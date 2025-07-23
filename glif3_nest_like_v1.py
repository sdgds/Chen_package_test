import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from spikingjelly.clock_driven import neuron, functional
from typing import Callable, Optional
from scipy.linalg import expm


class GLIF3Neuron(neuron.BaseNode):
    """
    GLIF3 (Generalized Leaky Integrate-and-Fire) neuron model
    Compatible with NEST glif_psc implementation with synaptic dynamics
    """
    
    def __init__(self, 
                 V_m: float = -70.0,           # Initial membrane potential (mV)
                 V_th: float = -50.0,          # Threshold potential (mV) 
                 g: float = 5.0,               # Leak conductance (nS)
                 E_L: float = -70.0,           # Leak reversal potential (mV)
                 C_m: float = 100.0,           # Membrane capacitance (pF)
                 t_ref: float = 2.0,           # Refractory period (ms)
                 V_reset: float = -70.0,       # Reset potential (mV)
                 asc_init: list = [0.0, 0.0], # Initial after-spike currents (pA)
                 asc_decay: list = [0.003, 0.1],  # After-spike current decay rates (1/ms)
                 asc_amps: list = [-10.0, -100.0], # After-spike current amplitudes (pA)
                 tau_syn: list = [5.5, 8.5, 2.8, 5.8], # Synaptic time constants (ms)
                 dt: float = 0.1,              # Time step (ms)
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self._validate_parameters(V_m, V_th, g, E_L, C_m, t_ref, dt, asc_decay, asc_amps, tau_syn)
        
        # Store parameters
        self.V_m_mV = V_m
        self.V_th_mV = V_th 
        self.g_nS = g
        self.E_L_mV = E_L
        self.C_m_pF = C_m
        self.t_ref_ms = t_ref
        self.V_reset_mV = V_reset
        self.dt_ms = dt
        
        # After-spike current parameters 
        self.asc_decay_rates = np.array(asc_decay, dtype=np.float64)
        self.asc_amps_pA = np.array(asc_amps, dtype=np.float64)
        self.num_asc = len(asc_decay)
        
        # Synaptic parameters
        self.tau_syn_ms = np.array(tau_syn, dtype=np.float64)
        self.syn_decay_rates = 1.0 / self.tau_syn_ms  # Convert to decay rates (1/ms)
        self.num_syn = len(tau_syn)
        
        # Derived parameters 
        self.tau_m_ms = C_m / g
        
        # Voltage normalization for SpikingJelly 
        self.v_range = self.V_th_mV - self.E_L_mV
        self.v_th_norm = 1.0
        self.v_reset_norm = (self.V_reset_mV - self.E_L_mV) / self.v_range
        self.e_l_norm = 0.0
        
        # Pre-computed tensor constants
        self.v_range_tensor = torch.tensor(self.v_range, dtype=torch.float32)
        self.e_l_tensor = torch.tensor(self.E_L_mV, dtype=torch.float32)
        self.v_th_norm_tensor = torch.tensor(self.v_th_norm, dtype=torch.float32)
        self.v_reset_norm_tensor = torch.tensor(self.v_reset_norm, dtype=torch.float32)
        self.dt_tensor = torch.tensor(self.dt_ms, dtype=torch.float32)
        self.t_ref_tensor = torch.tensor(self.t_ref_ms, dtype=torch.float32)
        self.zero_tensor = torch.tensor(0.0, dtype=torch.float32)
        self.asc_amps_tensor = torch.tensor(self.asc_amps_pA, dtype=torch.float32)
        
        # Initialize state variables
        self.v = torch.tensor((self.V_m_mV - self.E_L_mV) / self.v_range, dtype=torch.float32)
        
        # Register memory for ASC
        self.register_memory('asc_1', torch.tensor(asc_init[0], dtype=torch.float32))
        self.register_memory('asc_2', torch.tensor(asc_init[1], dtype=torch.float32))
        self.register_memory('ref_count', torch.tensor(0.0, dtype=torch.float32))
        
        # Register memory for synaptic currents
        for i in range(self.num_syn):
            self.register_memory(f'syn_{i}', torch.tensor(0.0, dtype=torch.float32))
        
        # Matrix exponential for exact integration
        # State vector: [asc_1, asc_2, syn_0, syn_1, syn_2, syn_3, V-E_L]
        state_dim = 2 + self.num_syn + 1  # ASC + synaptic + voltage
        A = np.zeros((state_dim, state_dim), dtype=np.float64)
        
        # ASC dynamics
        A[0, 0] = -self.asc_decay_rates[0]
        A[1, 1] = -self.asc_decay_rates[1]
        
        # Synaptic dynamics
        for i in range(self.num_syn):
            A[2+i, 2+i] = -self.syn_decay_rates[i]
        
        # Voltage dynamics (coupling from ASC and synaptic currents)
        v_idx = state_dim - 1
        A[v_idx, 0] = 1.0 / self.C_m_pF  # ASC_1 -> V
        A[v_idx, 1] = 1.0 / self.C_m_pF  # ASC_2 -> V
        for i in range(self.num_syn):
            A[v_idx, 2+i] = 1.0 / self.C_m_pF  # syn_i -> V
        A[v_idx, v_idx] = -1.0 / self.tau_m_ms  # leak
        
        self.exp_A_dt = torch.tensor(expm(A * self.dt_ms), dtype=torch.float32)
        
        # External current integration
        b = np.zeros(state_dim, dtype=np.float64)
        b[v_idx] = 1.0 / self.C_m_pF  # external current affects voltage
        self.B_integral = torch.tensor(self._compute_B_integral(A, b, self.dt_ms), dtype=torch.float32)
        
        # Synaptic input integration matrices (for each synapse type)
        self.syn_B_integrals = []
        for i in range(self.num_syn):
            b_syn = np.zeros(state_dim, dtype=np.float64)
            b_syn[2+i] = 1.0  # synaptic input affects corresponding synapse
            syn_B = torch.tensor(self._compute_B_integral(A, b_syn, self.dt_ms), dtype=torch.float32)
            self.syn_B_integrals.append(syn_B)
        
    def _validate_parameters(self, V_m, V_th, g, E_L, C_m, t_ref, dt, asc_decay, asc_amps, tau_syn):
        """Parameter validation"""
        if V_th <= V_m:
            raise ValueError(f"Threshold potential ({V_th}) must be greater than membrane potential ({V_m})")
        if g <= 0:
            raise ValueError(f"Leak conductance ({g}) must be positive")
        if C_m <= 0:
            raise ValueError(f"Membrane capacitance ({C_m}) must be positive")
        if t_ref < 0:
            raise ValueError(f"Refractory period ({t_ref}) cannot be negative")
        if dt <= 0 or dt > 1.0:
            raise ValueError(f"Time step ({dt}) must be in (0, 1] range")
        if len(asc_decay) != len(asc_amps):
            raise ValueError("ASC decay rates and amplitudes must have same length")
        if any(k <= 0 for k in asc_decay):
            raise ValueError("ASC decay rates must be positive")
        if any(tau <= 0 for tau in tau_syn):
            raise ValueError("Synaptic time constants must be positive")
            
    def _compute_B_integral(self, A, b, dt):
        """Compute integral for constant external input"""
        try:
            exp_A_dt = expm(A * dt)
            I = np.eye(A.shape[0])
            A_inv = np.linalg.pinv(A)
            return (exp_A_dt - I) @ A_inv @ b
        except np.linalg.LinAlgError:
            return b * dt
        
    def neuronal_charge(self, x: torch.Tensor, syn_inputs: Optional[torch.Tensor] = None):
        """Update state variables using exact integration
        
        Args:
            x: External current input (pA)
            syn_inputs: Synaptic inputs for each synapse type (pA), shape: (num_syn,)
        """
        not_refractory = self.ref_count <= self.zero_tensor
        
        # Build state vector: [asc_1, asc_2, syn_0, syn_1, syn_2, syn_3, V-E_L]
        v_mv = self.v * self.v_range_tensor + self.e_l_tensor
        state_list = [self.asc_1, self.asc_2]
        
        # Add synaptic currents to state vector
        for i in range(self.num_syn):
            state_list.append(getattr(self, f'syn_{i}'))
        
        # Add voltage
        state_list.append(v_mv - self.e_l_tensor)
        
        state_vec = torch.stack(state_list)
        
        # Exact integration: y_new = exp(A*dt) * y_old + B_integral * I_ext + sum(syn_B * syn_input)
        new_state = torch.mv(self.exp_A_dt, state_vec) + self.B_integral * x
        
        # Add synaptic inputs if provided
        if syn_inputs is not None:
            if len(syn_inputs) != self.num_syn:
                raise ValueError(f"Expected {self.num_syn} synaptic inputs, got {len(syn_inputs)}")
            for i, syn_input in enumerate(syn_inputs):
                new_state += self.syn_B_integrals[i] * syn_input
        
        # Update states (only if not refractory)
        new_asc_1 = torch.where(not_refractory, new_state[0], self.asc_1)
        new_asc_2 = torch.where(not_refractory, new_state[1], self.asc_2)
        
        # Update synaptic currents
        for i in range(self.num_syn):
            new_syn = torch.where(not_refractory, new_state[2+i], getattr(self, f'syn_{i}'))
            setattr(self, f'syn_{i}', new_syn)
        
        # Update voltage
        v_idx = 2 + self.num_syn
        new_v_mv = torch.where(not_refractory, new_state[v_idx] + self.e_l_tensor, v_mv)
        
        self.asc_1 = new_asc_1
        self.asc_2 = new_asc_2
        self.v = (new_v_mv - self.e_l_tensor) / self.v_range_tensor
        
        # Update refractory counter
        self.ref_count = torch.clamp(self.ref_count - self.dt_tensor, min=0.0)

    def neuronal_fire(self):
        """Check for spike generation"""
        eps = 1e-7
        return (self.v >= (self.v_th_norm_tensor - eps)) & (self.ref_count <= eps)

    def neuronal_reset(self, spike: torch.Tensor):
        """Reset on spike"""
        self.v = torch.where(spike, self.v_reset_norm_tensor, self.v)
        self.asc_1 = torch.where(spike, self.asc_1 + self.asc_amps_tensor[0], self.asc_1)
        self.asc_2 = torch.where(spike, self.asc_2 + self.asc_amps_tensor[1], self.asc_2)
        self.ref_count = torch.where(spike, self.t_ref_tensor, self.ref_count)

    def forward(self, x: torch.Tensor, syn_inputs: Optional[torch.Tensor] = None):
        """Forward pass with optional synaptic inputs
        
        Args:
            x: External current input (pA)  
            syn_inputs: Optional synaptic inputs for each synapse type (pA), shape: (num_syn,)
        """
        self.neuronal_charge(x, syn_inputs)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike.float()

    def get_state(self):
        """Get current neuron state including synaptic currents"""
        state = {
            'v': self.v.item(),
            'v_mV': (self.v * self.v_range + self.E_L_mV).item(),
            'asc_1': self.asc_1.item(),
            'asc_2': self.asc_2.item(),
            'ref_count': self.ref_count.item(),
            'total_asc': (self.asc_1 + self.asc_2).item()
        }
        
        # Add synaptic current states
        for i in range(self.num_syn):
            state[f'syn_{i}'] = getattr(self, f'syn_{i}').item()
        
        # Total synaptic current
        total_syn = sum(getattr(self, f'syn_{i}').item() for i in range(self.num_syn))
        state['total_syn'] = total_syn
        
        return state


def load_parameters(json_file):
    """Load neuron parameters from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)


def test_fi_curve(neuron, current_range, duration=1000, dt=0.1):
    """Test F-I curve by measuring firing rates at different currents"""
    results = []
    
    for i_inj in current_range:
        functional.reset_net(neuron)
        
        n_steps = int(duration / dt)
        spikes = []
        
        for t in range(n_steps):
            current = torch.tensor(i_inj, dtype=torch.float32)
            spike = neuron(current)
            spikes.append(spike.item())
        
        spike_count = sum(spikes)
        frequency = spike_count / (duration / 1000.0)  # Hz
        
        results.append({
            'current': i_inj,
            'frequency': frequency,
            'spike_count': spike_count
        })
        
        print(f"Current: {i_inj:6.1f} pA, Spikes: {int(spike_count):3d}, Frequency: {frequency:6.2f} Hz")
    
    return results


def main():
    """Test GLIF3 neuron model with synaptic dynamics"""
    print("Testing GLIF3 neuron model with synaptic dynamics")
    
    # Load parameters
    params = load_parameters('sj/483018019_glif_psc.json')
    
    # Create neuron
    neuron = GLIF3Neuron(
        V_m=params['V_m'],
        V_th=params['V_th'], 
        g=params['g'],
        E_L=params['E_L'],
        C_m=params['C_m'],
        t_ref=params['t_ref'],
        V_reset=params['V_reset'],
        asc_init=params['asc_init'],
        asc_decay=params['asc_decay'],
        asc_amps=params['asc_amps'],
        tau_syn=params['tau_syn'],
        dt=0.1
    )
    
    # Test F-I curve
    test_currents = range(0, 256, 15)
    print(f"\nF-I curve test:")
    
    fi_results = test_fi_curve(neuron, test_currents, duration=1000, dt=0.1)


if __name__ == "__main__":
    main() 

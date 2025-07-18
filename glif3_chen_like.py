import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from spikingjelly.clock_driven import neuron, functional
from typing import Callable, Optional


class GLIF3Neuron(neuron.BaseNode):
    """
    GLIF3 (Generalized Leaky Integrate-and-Fire) neuron model
    Implementation using separate integration method
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
                 dt: float = 0.1,              # Time step (ms)
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self._validate_parameters(V_m, V_th, g, E_L, C_m, t_ref, dt, asc_decay, asc_amps)
        
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
        self.asc_decay_rates = torch.tensor(asc_decay, dtype=torch.float32)
        self.asc_amps_pA = torch.tensor(asc_amps, dtype=torch.float32)
        self.num_asc = len(asc_decay)
        
        # Derived parameters 
        self.tau_m_ms = C_m / g
        
        # Voltage normalization for SpikingJelly
        self.v_range = self.V_th_mV - self.E_L_mV
        self.v_th_norm = 1.0
        self.v_reset_norm = (self.V_reset_mV - self.E_L_mV) / self.v_range
        self.e_l_norm = 0.0
        
        # Pre-computed tensor constants
        self.v_range_tensor = torch.tensor(self.v_range, dtype=torch.float32)
        self.e_l_norm_tensor = torch.tensor(self.e_l_norm, dtype=torch.float32)
        self.v_th_norm_tensor = torch.tensor(self.v_th_norm, dtype=torch.float32)
        self.v_reset_norm_tensor = torch.tensor(self.v_reset_norm, dtype=torch.float32)
        self.dt_tensor = torch.tensor(self.dt_ms, dtype=torch.float32)
        self.t_ref_tensor = torch.tensor(self.t_ref_ms, dtype=torch.float32)
        self.zero_tensor = torch.tensor(0.0, dtype=torch.float32)
        self.c_m_tensor = torch.tensor(self.C_m_pF, dtype=torch.float32)
        self.tau_m_c_m_v_range_tensor = torch.tensor(self.tau_m_ms / self.C_m_pF / self.v_range, dtype=torch.float32)
        
        # Initialize state variables
        self.v = torch.tensor((self.V_m_mV - self.E_L_mV) / self.v_range, dtype=torch.float32)
        
        # Register memory
        self.register_memory('asc_1', torch.tensor(asc_init[0], dtype=torch.float32))
        self.register_memory('asc_2', torch.tensor(asc_init[1], dtype=torch.float32))
        self.register_memory('ref_count', torch.tensor(0.0, dtype=torch.float32))
        
        # Pre-compute integration constants
        self.asc_exp_decay = torch.exp(-self.dt_tensor * self.asc_decay_rates)
        self.membrane_exp_decay = torch.exp(-self.dt_tensor / torch.tensor(self.tau_m_ms))
        
    def _validate_parameters(self, V_m, V_th, g, E_L, C_m, t_ref, dt, asc_decay, asc_amps):
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
        
    def neuronal_charge(self, x: torch.Tensor):
        """Update membrane potential and after-spike currents using separate integration"""
        not_refractory = self.ref_count <= self.zero_tensor
        
        # Update after-spike currents using exact integration
        self.asc_1 = self.asc_1 * self.asc_exp_decay[0]
        self.asc_2 = self.asc_2 * self.asc_exp_decay[1]
        
        # Total after-spike current
        total_asc_pA = self.asc_1 + self.asc_2
        
        # Current effects on voltage (pre-computed scaling)
        i_ext_effect = x / self.c_m_tensor * self.dt_tensor / self.v_range_tensor
        asc_effect = total_asc_pA / self.c_m_tensor * self.dt_tensor / self.v_range_tensor
        
        # Membrane voltage evolution using exact integration
        membrane_factor = (1.0 - self.membrane_exp_decay) * self.tau_m_c_m_v_range_tensor
        
        # Update voltage
        old_v_from_el = self.v - self.e_l_norm_tensor
        new_v_from_el = old_v_from_el * self.membrane_exp_decay
        current_contribution = (x + total_asc_pA) * membrane_factor
        
        self.v = torch.where(not_refractory, 
                           self.e_l_norm_tensor + new_v_from_el + current_contribution,
                           self.v)
        
        # Update refractory counter
        self.ref_count = torch.clamp(self.ref_count - self.dt_tensor, min=0.0)

    def neuronal_fire(self):
        """Check for spike generation"""
        eps = 1e-7
        return (self.v >= (self.v_th_norm_tensor - eps)) & (self.ref_count <= eps)

    def neuronal_reset(self, spike: torch.Tensor):
        """Reset on spike"""
        self.v = torch.where(spike, self.v_reset_norm_tensor, self.v)
        self.asc_1 = torch.where(spike, self.asc_1 + self.asc_amps_pA[0], self.asc_1)
        self.asc_2 = torch.where(spike, self.asc_2 + self.asc_amps_pA[1], self.asc_2)
        self.ref_count = torch.where(spike, self.t_ref_tensor, self.ref_count)

    def forward(self, x: torch.Tensor):
        """Forward pass"""
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike.float()

    def get_state(self):
        """Get current neuron state"""
        return {
            'v': self.v.item(),
            'v_mV': (self.v * self.v_range + self.E_L_mV).item(),
            'asc_1': self.asc_1.item(),
            'asc_2': self.asc_2.item(),
            'ref_count': self.ref_count.item(),
            'total_asc': (self.asc_1 + self.asc_2).item()
        }


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
    """Test GLIF3 neuron model"""
    print("Testing GLIF3 neuron model (separate integration)")
    
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
        dt=0.1
    )
    
    # Test F-I curve
    test_currents = range(0, 256, 15)
    print(f"\nF-I curve test:")
    
    fi_results = test_fi_curve(neuron, test_currents, duration=1000, dt=0.1)


if __name__ == "__main__":
    main() 
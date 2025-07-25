#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经元响应分析脚本
该脚本用于分析不同神经元类型在不同平台电流强度下的膜电位和脉冲响应。
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
from tqdm import tqdm
from scipy import stats
from scipy.signal import correlate
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')


def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude

@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad

class SingleNeuronModel(tf.keras.layers.Layer):
    def __init__(self, neuron_model_template_index, 
                       model_path='../GLIF_network/network_dat.pkl', n_neurons=1,
                       dt=1., gauss_std=.5, dampening_factor=.3,
                       input_weight_scale=1., recurrent_weight_scale=1.,
                       spike_gradient=False, max_delay=5, train_recurrent=True, train_input=True, 
                       train_bkg=False, use_dale_law=False, _return_interal_variables=False):
        super().__init__()
        network = self.build_single_type_network(neuron_model_template_index, model_path, n_neurons)
        input_population, bkg_weights = self.build_dummy_input(n_neurons)
        self._params = network['node_params']
        self._compute_dtype = tf.float32

        voltage_scale = self._params['V_th'] - self._params['E_L']
        voltage_offset = self._params['E_L']
        self._params['V_th'] = (self._params['V_th'] - voltage_offset) / voltage_scale
        self._params['E_L'] = (self._params['E_L'] - voltage_offset) / voltage_scale
        self._params['V_reset'] = (self._params['V_reset'] - voltage_offset) / voltage_scale
        self._params['asc_amps'] = self._params['asc_amps'] / voltage_scale[..., None]

        self._node_type_ids = network['node_type_ids']
        self._dt = dt   

        self._return_interal_variables = _return_interal_variables

        # for random spike, the instantaneous firing rate when v = v_th
        self._spike_gradient = spike_gradient

        n_receptors = network['node_params']['tau_syn'].shape[1]
        self._n_receptors = n_receptors
        self._n_neurons = network['n_nodes']
        self._dampening_factor = tf.cast(dampening_factor, self._compute_dtype)
        self._gauss_std = tf.cast(gauss_std, self._compute_dtype)

        tau = self._params['C_m'] / self._params['g']
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / self._params['C_m'] * (1 - self._decay) * tau
        self._syn_decay = np.zeros(self._params['tau_syn'].shape)
        self._psc_initial = np.zeros(self._params['tau_syn'].shape)

        # synapses: target_ids, source_ids, weights, delays
        max_delay_ms = np.min([np.max(network['synapses']['delays']), max_delay])
        self.max_delay = int(np.round(max_delay_ms / dt))

        self.state_size = (
            self._n_neurons * self.max_delay,  # z buffer
            self._n_neurons,                   # v
            self._n_neurons,                   # r
            self._n_neurons,                   # asc 1
            self._n_neurons,                   # asc 2
            n_receptors * self._n_neurons,     # psc rise
            n_receptors * self._n_neurons,     # psc
        )
        # useless now; it was for training the neuron parameters
        def _f(_v, trainable=False):
            return tf.Variable(tf.cast(self._gather(_v), self._compute_dtype), trainable=trainable)

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        # useless
        def custom_val(_v, trainable=False):
            _v = tf.Variable(tf.cast(inv_sigmoid(self._gather(_v)), self._compute_dtype), trainable=trainable)

            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        self.v_reset = _f(self._params['V_reset'])
        self.syn_decay = _f(self._syn_decay)
        self.psc_initial = _f(self._psc_initial)
        self.t_ref = _f(self._params['t_ref'])
        self.asc_amps = _f(self._params['asc_amps'], trainable=False)
        # self.param_k = _f(self._params['k'], trainable=True)
        _k = self._params['k']
        # _k[_k < .0031] = .0007
        self.param_k, self.param_k_read = custom_val(_k, trainable=False)
        self.v_th = _f(self._params['V_th'])
        self.e_l = _f(self._params['E_L'])
        self.param_g = _f(self._params['g'])
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)

        self.recurrent_weights = None
        self.disconnect_mask = None

        indices, weights, dense_shape = \
            network['synapses']['indices'], network['synapses']['weights'], network['synapses']['dense_shape']
        weights = weights / voltage_scale[self._node_type_ids[indices[:, 0] // self._n_receptors]]
        # 🔥 关键修改：使用以毫秒为单位的max_delay进行clipping
        delays = np.round(np.clip(network['synapses']['delays'], dt, max_delay_ms) / dt).astype(np.int32)
        dense_shape = dense_shape[0], self.max_delay * dense_shape[1]
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
        weights = weights.astype(np.float32)
        print(f'> Recurrent synapses {len(indices)}')
        input_weights = input_population['weights'].astype(np.float32)
        input_indices = input_population['indices']
        input_weights = input_weights / voltage_scale[self._node_type_ids[input_indices[:, 0] // self._n_receptors]]
        print(f'> Input synapses {len(input_indices)}')
        input_dense_shape = (self._n_receptors * self._n_neurons, input_population['n_inputs'])

        self.recurrent_weight_positive = tf.Variable(
            weights >= 0., name='recurrent_weights_sign', trainable=False)
        self.input_weight_positive = tf.Variable(
            input_weights >= 0., name='input_weights_sign', trainable=False)
        if use_dale_law:
            self.recurrent_weight_values = tf.Variable(
                weights * recurrent_weight_scale, name='sparse_recurrent_weights',
                constraint=SignedConstraint(self.recurrent_weight_positive),
                trainable=train_recurrent)
        else:
            self.recurrent_weight_values = tf.Variable(
                weights * recurrent_weight_scale, name='sparse_recurrent_weights',
                constraint=None,
                trainable=train_recurrent)
        self.recurrent_indices = tf.Variable(indices, trainable=False)
        self.recurrent_dense_shape = dense_shape

        if use_dale_law:
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale, name='sparse_input_weights',
                constraint=SignedConstraint(self.input_weight_positive),
                trainable=train_input)
        else:
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale, name='sparse_input_weights',
                constraint=None,
                trainable=train_input)

        self.input_indices = tf.Variable(input_indices, trainable=False)
        self.input_dense_shape = input_dense_shape
        bkg_weights = bkg_weights / np.repeat(voltage_scale[self._node_type_ids], self._n_receptors)
        # this actutually is not used; we used the decoded noise
        self.bkg_weights = tf.Variable(bkg_weights * 10., name='rest_of_brain_weights', trainable=train_bkg)

    def build_single_type_network(self, neuron_model_template_index, 
                                  model_path, n_neurons):
        """ 目的是创建最小、必要的网络结构 """
        with open(model_path, 'rb') as f:
            d = pkl.load(f)
        node_params = {k: np.array([v]) for k, v in d['nodes'][neuron_model_template_index]['params'].items()}

        # 获取n_receptors
        n_receptors = node_params['tau_syn'].shape[1]

        # 添加虚拟突触连接（例如自突触）
        indices = np.array([[0, 0]], dtype=np.int64)
        weights = np.array([0.0], dtype=np.float32)   # 定义为零
        delays = np.array([4.0], dtype=np.float32)

        return dict(
            node_params=node_params,
            node_type_ids=np.zeros(n_neurons, dtype=np.int64),
            n_nodes=n_neurons,
            synapses=dict(
                indices=indices,
                weights=weights,
                delays=delays,
                dense_shape=(n_receptors * n_neurons, n_neurons)
            )
        )

    def build_dummy_input(self, n_neurons):
        input_population = dict(
            indices=np.zeros((0, 2), dtype=np.int64),
            weights=np.zeros(0, dtype=np.float32),
            n_inputs=1
        )
        bkg_weights = np.zeros(n_neurons * 4, dtype=np.float32)
        return input_population, bkg_weights

    def zero_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * tf.cast(self.v_th * .0 + 1. * self.v_reset, dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_20 = tf.zeros((batch_size, self._n_neurons), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_receptors), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_receptors), dtype)
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def random_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.cast(tf.random.uniform((batch_size, self._n_neurons * self.max_delay), 0, 2, tf.int32), dtype)
        v0 = tf.random.uniform((batch_size, self._n_neurons), tf.cast(self.v_reset,dtype), tf.cast(self.v_th,dtype), dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.random.normal((batch_size, self._n_neurons), mean=-0.28, stddev=1.75, dtype=dtype) # min -87 max 59
        asc_20 = tf.random.normal((batch_size, self._n_neurons), mean=-0.28, stddev=1.75, dtype=dtype)
        psc_rise0 = tf.random.normal((batch_size, self._n_neurons * self._n_receptors), mean=0.29, stddev=0.77, dtype=dtype) #-3.8~33.6
        psc0 = tf.random.normal((batch_size, self._n_neurons * self._n_receptors), mean=1.17, stddev=3.19, dtype=dtype) # -21~147
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def _gather(self, prop):
        return tf.gather(prop, self._node_type_ids)

    def call(self, inputs, state):
        batch_size = 1
           
        z_buf, v, r, asc_1, asc_2, psc_rise, psc = state

        shaped_z_buf = tf.reshape(z_buf, (-1, self.max_delay, self._n_neurons))
        prev_z = shaped_z_buf[:, 0]

        psc_rise = tf.reshape(psc_rise, (batch_size, self._n_neurons, self._n_receptors))
        psc = tf.reshape(psc, (batch_size, self._n_neurons, self._n_receptors))

        sparse_w_rec = tf.sparse.SparseTensor(
            self.recurrent_indices, self.recurrent_weight_values, self.recurrent_dense_shape)

        i_rec = tf.sparse.sparse_dense_matmul(sparse_w_rec, tf.cast(z_buf, tf.float32), adjoint_b=True)
        i_rec = tf.transpose(i_rec)

        rec_inputs = tf.cast(i_rec, self._compute_dtype)
        # 🔥 关键修改：循环电流去除inputs，inputs不应该作为突触电流注入，而应该直接注入在神经元上
        rec_inputs = tf.reshape(rec_inputs, (batch_size, self._n_neurons, self._n_receptors))        

        new_psc_rise = self.syn_decay * psc_rise + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise

        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)

        k = self.param_k_read()
        asc_amps = self.asc_amps
        new_asc_1 = tf.exp(-self._dt * k[:, 0]) * asc_1 + prev_z * asc_amps[:, 0]
        new_asc_2 = tf.exp(-self._dt * k[:, 1]) * asc_2 + prev_z * asc_amps[:, 1]

        reset_current = prev_z * (self.v_reset - self.v_th)
        input_current = tf.reduce_sum(psc, -1)
        decayed_v = self.decay * v

        # 🔥 关键修改：正确的直接电流注入（包含标准化）
        gathered_g = self.param_g * self.e_l
        external_current = inputs / self.voltage_scale
        c1 = input_current + asc_1 + asc_2 + gathered_g + external_current
        new_v = decayed_v + self.current_factor * c1 + reset_current

        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th) / normalizer
        
        new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)        

        new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z)

        new_psc = tf.reshape(new_psc, (batch_size, self._n_neurons * self._n_receptors))
        new_psc_rise = tf.reshape(new_psc_rise, (batch_size, self._n_neurons * self._n_receptors))

        new_shaped_z_buf = tf.concat((new_z[:, None], shaped_z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_shaped_z_buf, (-1, self._n_neurons * self.max_delay))

        if self._return_interal_variables:
            new_ascs = tf.concat((new_asc_1, new_asc_2), -1)
            outputs = (new_z, new_v * self.voltage_scale + self.voltage_offset, new_ascs, new_psc_rise, new_psc)
        else:
            outputs = (new_z, new_v * self.voltage_scale + self.voltage_offset)
        new_state = (new_z_buf, new_v, new_r, new_asc_1, new_asc_2, new_psc_rise, new_psc)

        return outputs, new_state

    def simulate(self, T, platform_current, current_start, current_end):
        """
        模拟一段时间
        
        参数:
        T: 模拟时长 (ms)
        platform_current: 平台电流 (pA)
        current_start: 电流开始时间 (ms)
        current_end: 电流结束时间 (ms)
        
        返回:
        time: 时间数组
        current_sequence: 电流序列
        voltages: 电压数组
        spikes: spike数组
        """
        n_steps = int(T / self._dt)
        time = np.arange(n_steps) * self._dt
        
        # 构建输入电流序列 - 修复：使用n_steps而不是T
        current_sequence = np.zeros(n_steps, dtype=np.float32)
        
        # 计算电流开始和结束的时间步索引
        current_start_step = int(current_start / self._dt)
        current_end_step = int(current_end / self._dt)
        
        # 确保索引在有效范围内
        current_start_step = max(0, min(current_start_step, n_steps))
        current_end_step = max(0, min(current_end_step, n_steps))
        
        current_sequence[current_start_step:current_end_step] = platform_current
        current = tf.convert_to_tensor(current_sequence)
            
        spikes = np.zeros(n_steps)
        voltages = np.zeros(n_steps)
        
        state = self.zero_state(batch_size=1)
        for i in range(n_steps):
            (spike, voltage), state = self(current[i], state)
            spikes[i] = spike
            voltages[i] = voltage
            
        return time, current_sequence, voltages, spikes


def plot_single_response(time, current, voltage, spikes, neuron_type, current_amplitude):
    """
    绘制单个神经元的响应图
    
    参数:
    - time: 时间序列
    - current: 输入电流序列
    - voltage: 膜电位序列
    - spikes: 脉冲序列
    - neuron_type: 神经元类型
    - current_amplitude: 电流幅度
    - save_path: 保存路径 (可选)
    - show_plot: 是否显示图像
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,4), sharex=True)
    
    # 上方子图：输入电流
    ax1.plot(time, current, 'b-', linewidth=2, label=f'Input current ({current_amplitude} pA)')
    ax1.set_ylabel('Current (pA)', fontsize=12)
    ax1.set_title(f'{neuron_type} neuron response - current: {current_amplitude} pA', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 下方子图：膜电位和脉冲
    ax2.plot(time, voltage, 'k-', linewidth=1.5, label='Membrane potential')
    
    # 添加脉冲标记（在对应时间点的膜电位上画红色竖线）
    spike_indices = np.where(spikes > 0.5)[0]  # 找到脉冲发生的时间点索引
    for spike_idx in spike_indices:
        spike_voltage = voltage[spike_idx]  # 获取脉冲时刻的膜电位值
        ax2.plot(time[spike_idx], spike_voltage, '|k', markersize=10, color='red')
    
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Membrane potential (mV)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加脉冲计数信息
    spike_count = np.sum(spikes > 0.5)
    firing_rate = spike_count / ( (np.where(current!=0)[0].shape[0]) / 1000)  # Hz
    ax2.text(0.02, 0.98, f'Spike number: {spike_count}\n firing rate: {firing_rate:.2f} Hz', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_current_response(neuron_type, neuron, T, current_amplitudes, current_start, current_end, plot=False):
    """
    分析神经元在不同电流强度下的响应
    
    参数:
    - neuron_type: 神经元类型
    - current_amplitudes: 电流强度列表
    - model_path: 模型文件路径
    - show_plot: 是否显示图像
    """
    
    n_currents = len(current_amplitudes)
    if plot==True:
        # 创建多子图
        fig, axes = plt.subplots(n_currents + 1, 1, figsize=(10, 2 * (n_currents + 1)), 
                                gridspec_kw={'hspace': 0.3})
        
        # 颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, n_currents))
        
        # 第一个子图：显示所有电流波形
        time_ref = np.arange(T)  # 参考时间序列
        for i, current_amp in enumerate(current_amplitudes):
            current_waveform = np.zeros(T)
            current_waveform[current_start:current_end] = current_amp
            axes[0].plot(time_ref, current_waveform, color=colors[i], linewidth=2, 
                        label=f'{current_amp} pA')
        
        axes[0].set_ylabel('Current (pA)', fontsize=12)
        axes[0].set_title(f'{neuron_type} neuron response', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 存储结果用于分析
    all_results = []
    
    # 对每个电流强度进行仿真
    for i, current_amp in enumerate(tqdm(current_amplitudes)):        
        # 运行仿真
        time, current, voltage, spikes = neuron.simulate(
            T, current_amp, current_start, current_end
        )
        
        # 存储结果
        spike_count = np.sum(spikes > 0.5)
        firing_rate = spike_count / ( (np.where(current!=0)[0].shape[0]) / 1000)
        all_results.append({
            'current': current_amp,
            'spike_count': spike_count,
            'firing_rate': firing_rate,
            'voltage': voltage,
            'spikes': spikes
        })
        
        if plot==True:
        # 绘制到对应子图
            ax = axes[i + 1]
            ax.plot(time, voltage, color='black', linewidth=1.5, label='Membrane potential')
            
            # 添加脉冲标记（在对应时间点的膜电位上画红色竖线）
            spike_indices = np.where(spikes > 0.5)[0]
            for spike_idx in spike_indices:
                spike_voltage = voltage[spike_idx]  # 获取脉冲时刻的膜电位值
                ax.plot(time[spike_idx], spike_voltage, '|k', markersize=10, color='red')
            
            ax.set_ylabel('Membrane potential (mV)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.98, f'{current_amp} nA\n spike: {spike_count}\n rate: {firing_rate:.1f} Hz', 
                    transform=ax.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if plot==True:
        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        plt.tight_layout(pad=0.5)
        plt.show()
    
    return all_results

def plot_if_curve(neuron_type, results, save_dir=None):
    """
    绘制电流-发放频率(I-F)曲线
    
    参数:
    - neuron_type: 神经元类型
    - results: 仿真结果列表
    """
    
    currents = [r['current'] for r in results]
    firing_rates = [r['firing_rate'] for r in results]
    
    plt.figure(figsize=(9,3))
    plt.plot(currents, firing_rates, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('input current (nA)', fontsize=12)
    plt.ylabel('firing rate (Hz)', fontsize=12)
    plt.title(f'{neuron_type} I-F curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加数据标签
    for i, (curr, rate) in enumerate(zip(currents, firing_rates)):
        plt.annotate(f'{rate:.1f}', (curr, rate), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir + '/' + f'{neuron_type}_if_curve.png', dpi=100)
        
    plt.show()


class SpikeTrainSynchronization:
    """发放序列同步性分析类"""
    
    def __init__(self, dt=1.0):
        """
        初始化
        
        Args:
            dt: 时间步长（ms）
        """
        self.dt = dt
    
    def get_spike_times(self, spike_train):
        """
        从发放序列中提取发放时间点
        
        Args:
            spike_train: 发放序列（0/1数组）
        
        Returns:
            spike_times: 发放时间点数组（ms）
        """
        return np.where(spike_train > 0)[0] * self.dt
    
    def cross_correlation_histogram(self, spike_train1, spike_train2, 
                                   max_lag=100, bin_width=1.0, normalize=True):
        """
        计算交叉相关直方图（Cross-Correlation Histogram, CCH）
        
        Args:
            spike_train1, spike_train2: 发放序列
            max_lag: 最大时间延迟（ms）
            bin_width: 直方图bin宽度（ms）
            normalize: 是否归一化
        
        Returns:
            cch: 交叉相关直方图
            lags: 延迟时间轴
            peak_amplitude: 峰值幅度
            fwhm: 半高宽（同步精度）
        """
        # 获取发放时间点
        spike_times1 = self.get_spike_times(spike_train1)
        spike_times2 = self.get_spike_times(spike_train2)
        
        if len(spike_times1) == 0 or len(spike_times2) == 0:
            return np.array([]), np.array([]), 0, 0
        
        # 计算所有时间差
        time_diffs = []
        for t1 in spike_times1:
            for t2 in spike_times2:
                diff = t1 - t2
                if abs(diff) <= max_lag:
                    time_diffs.append(diff)
        
        if len(time_diffs) == 0:
            return np.array([]), np.array([]), 0, 0
        
        # 创建直方图
        bins = np.arange(-max_lag, max_lag + bin_width, bin_width)
        cch, bin_edges = np.histogram(time_diffs, bins=bins)
        lags = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 归一化
        if normalize:
            # 使用发放率归一化
            rate1 = len(spike_times1) / (len(spike_train1) * self.dt / 1000.0)  # Hz
            rate2 = len(spike_times2) / (len(spike_train2) * self.dt / 1000.0)  # Hz
            expected_coincidences = rate1 * rate2 * len(spike_train1) * self.dt / 1000.0 * bin_width / 1000.0
            if expected_coincidences > 0:
                cch = cch / expected_coincidences
        
        # 计算峰值幅度（lag=0附近的平均值，严格符合文献）
        center_mask = np.abs(lags) <= bin_width  # 例如bin_width=1ms时，取-1,0,1ms
        sync_peak = np.mean(cch[center_mask])
        
        # 计算半高宽（FWHM，以同步峰值为基准）
        half_max = sync_peak / 2
        above_half = cch >= half_max
        if np.any(above_half):
            fwhm = np.sum(above_half) * bin_width
        else:
            fwhm = 0
        
        return cch, lags, sync_peak, fwhm
    
    def mutual_information(self, spike_train1, spike_train2, 
                          window_size=50, n_bins=10):
        """
        计算互信息（Mutual Information, MI）
        
        Args:
            spike_train1, spike_train2: 发放序列
            window_size: 滑动窗口大小（ms）
            n_bins: 直方图bin数量
        
        Returns:
            mi: 互信息值
            mi_normalized: 归一化互信息值
        """
        # 确保长度一致
        min_length = min(len(spike_train1), len(spike_train2))
        spike_train1 = spike_train1[:min_length]
        spike_train2 = spike_train2[:min_length]

        # 使用直方图方法
        # 计算滑动窗口发放率
        rates1 = []
        rates2 = []
        
        for i in range(0, min_length - window_size, window_size):
            window1 = spike_train1[i:i+window_size]
            window2 = spike_train2[i:i+window_size]
            
            rate1 = np.sum(window1) / (window_size * self.dt / 1000.0)  # Hz
            rate2 = np.sum(window2) / (window_size * self.dt / 1000.0)  # Hz
            
            rates1.append(rate1)
            rates2.append(rate2)
        
        rates1 = np.array(rates1)
        rates2 = np.array(rates2)
        
        # 创建直方图
        hist1, _ = np.histogram(rates1, bins=n_bins)
        hist2, _ = np.histogram(rates2, bins=n_bins)
        
        # 计算联合直方图
        joint_hist, _, _ = np.histogram2d(rates1, rates2, bins=n_bins)
        
        # 计算互信息
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_hist[i, j] > 0 and hist1[i] > 0 and hist2[j] > 0:
                    p_joint = joint_hist[i, j] / np.sum(joint_hist)
                    p1 = hist1[i] / np.sum(hist1)
                    p2 = hist2[j] / np.sum(hist2)
                    mi += p_joint * np.log2(p_joint / (p1 * p2))
        
        # 归一化互信息（除以最小熵）
        entropy1 = stats.entropy(hist1)
        entropy2 = stats.entropy(hist2)
        min_entropy = min(entropy1, entropy2)
        
        mi_normalized = mi / min_entropy if min_entropy > 0 else 0
        
        return mi, mi_normalized
    
    def coincidence_firing_rate(self, spike_train1, spike_train2, 
                               time_tolerance=5.0, correct_random=True):
        """
        计算协发放率（Coincidence Firing Rate）
        
        Args:
            spike_train1, spike_train2: 发放序列
            time_tolerance: 时间容差（ms）
            correct_random: 是否矫正随机性
        
        Returns:
            coincidence_rate: 协发放率（Hz）
            coincidence_count: 协发放次数
            expected_random: 期望随机协发放次数
            corrected_rate: 矫正后的协发放率
        """
        # 获取发放时间点
        spike_times1 = self.get_spike_times(spike_train1)
        spike_times2 = self.get_spike_times(spike_train2)
        
        if len(spike_times1) == 0 or len(spike_times2) == 0:
            return 0, 0, 0, 0
        
        # 计算协发放次数
        coincidence_count = 0
        for t1 in spike_times1:
            for t2 in spike_times2:
                if abs(t1 - t2) <= time_tolerance:
                    coincidence_count += 1
        
        # 计算协发放率
        total_time = len(spike_train1) * self.dt / 1000.0  # 秒
        coincidence_rate = coincidence_count / total_time  # Hz
        
        # 计算期望随机协发放次数
        if correct_random:
            rate1 = len(spike_times1) / total_time  # Hz
            rate2 = len(spike_times2) / total_time  # Hz
            expected_random = rate1 * rate2 * total_time * (2 * time_tolerance / 1000.0)
            
            # 矫正后的协发放率
            corrected_count = coincidence_count - expected_random
            corrected_rate = corrected_count / total_time if corrected_count > 0 else 0
        else:
            expected_random = 0
            corrected_rate = coincidence_rate
        
        return coincidence_rate, coincidence_count, expected_random, corrected_rate
    
    def comprehensive_analysis(self, spike_train1, spike_train2, plot_synchronization=False,
                               neuron_names=["Neuron 1", "Neuron 2"]):
        """
        综合同步性分析
        
        Args:
            spike_train1, spike_train2: 发放序列
            neuron_names: 神经元名称
        
        Returns:
            results: 分析结果字典
        """
        print(f"=== {neuron_names[0]} vs {neuron_names[1]} Synchronization Analysis ===")
        
        # 1. 交叉相关直方图
        cch, lags, peak_amp, fwhm = self.cross_correlation_histogram(
            spike_train1, spike_train2, max_lag=50, bin_width=1.0)
        
        # 2. 互信息
        mi, mi_norm = self.mutual_information(spike_train1, spike_train2, 
                                             window_size=50, n_bins=10)
        
        # 3. 协发放率
        for tolerance in [2, 5, 10]:
            co_rate, co_count, exp_random, corr_rate = self.coincidence_firing_rate(
                spike_train1, spike_train2, time_tolerance=tolerance, correct_random=True)
        
        if plot_synchronization:
            self.plot_synchronization_analysis(spike_train1, spike_train2, 
                                            cch, lags, neuron_names)
        
        return {
            'cch': {'histogram': cch, 'lags': lags, 'peak_amplitude': peak_amp, 'fwhm': fwhm},
            'mutual_information': {'mi': mi, 'mi_normalized': mi_norm},
            'coincidence_rate': {
                'tolerance_2ms': self.coincidence_firing_rate(spike_train1, spike_train2, 2),
                'tolerance_5ms': self.coincidence_firing_rate(spike_train1, spike_train2, 5),
                'tolerance_10ms': self.coincidence_firing_rate(spike_train1, spike_train2, 10)
            }
        }
    
    def plot_synchronization_analysis(self, spike_train1, spike_train2, 
                                    cch, lags, neuron_names):
        """绘制同步性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Spike Trains
        time_axis = np.arange(len(spike_train1)) * self.dt
        axes[0, 0].plot(time_axis, spike_train1, 'b-', label=neuron_names[0], linewidth=0.8)
        axes[0, 0].plot(time_axis, spike_train2, 'r-', label=neuron_names[1], linewidth=0.8)
        axes[0, 0].set_title('Spike Trains')
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Spikes')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cross-Correlation Histogram (CCH)
        if len(cch) > 0:
            axes[0, 1].bar(lags, cch, width=1.0, alpha=0.7, color='green')
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Lag')
            axes[0, 1].set_title('Cross-Correlation Histogram (CCH)')
            axes[0, 1].set_xlabel('Time Lag (ms)')
            axes[0, 1].set_ylabel('Coincidence Count')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative Spike Count
        cumulative1 = np.cumsum(spike_train1)
        cumulative2 = np.cumsum(spike_train2)
        axes[1, 0].plot(time_axis, cumulative1, 'b-', label=neuron_names[0], linewidth=0.8)
        axes[1, 0].plot(time_axis, cumulative2, 'r-', label=neuron_names[1], linewidth=0.8)
        axes[1, 0].set_title('Cumulative Spike Count')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Cumulative Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Synchronization Metrics Summary
        mi, mi_norm = self.mutual_information(spike_train1, spike_train2)
        co_rate_2, _, _, corr_rate_2 = self.coincidence_firing_rate(spike_train1, spike_train2, 2)
        co_rate_5, _, _, corr_rate_5 = self.coincidence_firing_rate(spike_train1, spike_train2, 5)
        
        metrics = ['Normalized MI', '2ms Coincidence', '5ms Coincidence', 'CCH Peak']
        values = [mi_norm, corr_rate_2, corr_rate_5, 
                cch[np.argmax(cch)] if len(cch) > 0 else 0]
        colors = ['blue', 'green', 'orange', 'red']
        
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Synchronization Metrics Summary')
        axes[1, 1].set_ylabel('Synchronization Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
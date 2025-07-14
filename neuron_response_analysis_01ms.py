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


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    """高斯伪导数的spike函数"""
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude
        de_dv_scaled = de_dz * dz_dv_scaled
        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad


class SingleNeuronModel:
    """
    基于BillehColumn动力学的单神经元模型
    完全保持相同的动力学机制，但简化为单个神经元
    """
    
    def __init__(self, neuron_model_template_index, 
                       model_path='../GLIF_network/network_dat.pkl',
                       dt=0.1, gauss_std=0.5, dampening_factor=0.3):
        """
        初始化单神经元模型
        
        参数:
        neuron_model_template_index: 神经元模板索引
        model_path: 模型文件路径
        dt: 时间步长 (ms) - 与NEST保持一致，默认0.1ms以提高精度
        gauss_std: 高斯伪导数标准差
        dampening_factor: 阻尼因子
        """
        self._dt = dt
        self._gauss_std = gauss_std
        self._dampening_factor = dampening_factor
        
        # 读取模型文件
        with open(model_path, 'rb') as f:
            d = pkl.load(f)
        params = {k: np.array([v]) for k, v in d['nodes'][neuron_model_template_index]['params'].items()}

        # 复制原始参数 (与BillehColumn完全一致的处理)
        self._params = {}
        for key, value in params.items():
            self._params[key] = value.copy()
        
        # 电压归一化 (与BillehColumn完全一致)
        voltage_scale = self._params['V_th'] - self._params['E_L']
        voltage_offset = self._params['E_L']
        self._params['V_th'] = (self._params['V_th'] - voltage_offset) / voltage_scale
        self._params['E_L'] = (self._params['E_L'] - voltage_offset) / voltage_scale
        self._params['V_reset'] = (self._params['V_reset'] - voltage_offset) / voltage_scale
        self._params['asc_amps'] = self._params['asc_amps'] / voltage_scale[..., None]
        
        # 计算衍生参数 (与BillehColumn完全一致)
        tau = self._params['C_m'] / self._params['g']
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / self._params['C_m'] * (1 - self._decay) * tau
        self._syn_decay = np.exp(-dt / np.array(self._params['tau_syn']))
        self._psc_initial = np.e / np.array(self._params['tau_syn'])
        
        # 保存缩放参数
        self.voltage_scale = voltage_scale[0]
        self.voltage_offset = voltage_offset[0]
        
        # 转换为标量参数 (单神经元)
        self.v_th = self._params['V_th'][0]
        self.e_l = self._params['E_L'][0]
        self.v_reset = self._params['V_reset'][0]
        self.param_g = self._params['g'][0]
        self.t_ref = self._params['t_ref'][0]
        self.k = self._params['k'][0]
        self.asc_amps = self._params['asc_amps'][0]
        self.decay = self._decay[0]
        self.current_factor = self._current_factor[0]
        self.syn_decay = self._syn_decay[0]
        self.psc_initial = self._psc_initial[0]
        
        # 突触数量
        self.n_receptors = self._params['tau_syn'].shape[1]
        
        # 初始化状态
        self.reset_state()
    
    def reset_state(self):
        """重置神经元状态 (与BillehColumn的zero_state一致)"""
        self.v = self.v_th * 0.0 + 1.0 * self.v_reset  # 归一化电压
        self.r = 0.0  # 不应期计数器
        self.asc_1 = 0.0  # adaptation电流1
        self.asc_2 = 0.0  # adaptation电流2
        self.psc_rise = np.zeros(self.n_receptors)  # 突触电流上升
        self.psc = np.zeros(self.n_receptors)  # 突触电流
    
    def step(self, input_current):
        """
        执行一个时间步 (与BillehColumn的call方法完全一致)
        
        参数:
        input_current: 输入电流 (pA)
        
        返回:
        spike: 是否产生spike (0或1)
        voltage: 实际电压值 (mV)
        """
        # 保存上一步的spike状态
        prev_z = getattr(self, 'prev_z', 0)
        
        # 更新突触电流 (没有外部突触输入，只有递归连接)
        # 对于单神经元，没有递归连接，所以突触输入为0
        rec_inputs = np.zeros(self.n_receptors)
        
        # 完全按照BillehColumn的顺序更新突触电流
        new_psc_rise = self.syn_decay * self.psc_rise + rec_inputs * self.psc_initial
        new_psc = self.psc * self.syn_decay + self._dt * self.syn_decay * self.psc_rise
        
        # 更新不应期计数器 (与BillehColumn完全一致: tf.nn.relu)
        new_r = max(0, self.r + prev_z * self.t_ref - self._dt)
        
        # 更新adaptation电流 (与BillehColumn完全一致)
        new_asc_1 = np.exp(-self._dt * self.k[0]) * self.asc_1 + prev_z * self.asc_amps[0]
        new_asc_2 = np.exp(-self._dt * self.k[1]) * self.asc_2 + prev_z * self.asc_amps[1]
        
        # 计算电压 (与BillehColumn完全一致，但加上外部输入电流)
        reset_current = prev_z * (self.v_reset - self.v_th)
        input_current_sum = np.sum(self.psc)  # 注意：使用当前时刻的psc而不是new_psc
        decayed_v = self.decay * self.v
        
        gathered_g = self.param_g * self.e_l
        # 将外部输入电流(pA)转换为与模型一致的单位
        # 输入电流需要除以voltage_scale来归一化
        external_current = input_current / self.voltage_scale
        c1 = input_current_sum + self.asc_1 + self.asc_2 + gathered_g + external_current
        new_v = decayed_v + self.current_factor * c1 + reset_current
        
        # 检查是否产生spike (与BillehColumn完全一致)
        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th) / normalizer
        
        # 使用spike_gauss函数的逻辑 (与BillehColumn完全一致)
        # new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)        
        # new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z)
        new_z = 1.0 if new_v >= self.v_th and new_r <= 0 else 0.0
        
        # 更新状态
        self.psc_rise = new_psc_rise
        self.psc = new_psc
        self.r = new_r
        self.asc_1 = new_asc_1
        self.asc_2 = new_asc_2
        self.v = new_v
        
        # 保存当前spike状态供下一步使用
        self.prev_z = new_z
        
        return new_z, new_v * self.voltage_scale + self.voltage_offset
    
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
        
        # 构建输入电流序列 - 修复：基于时间步数而不是毫秒数
        current_sequence = np.zeros(n_steps, dtype=np.float32)
        
        # 将时间转换为步数索引
        start_step = int(current_start / self._dt)
        end_step = int(current_end / self._dt)
        
        # 确保索引在有效范围内
        start_step = max(0, min(start_step, n_steps))
        end_step = max(0, min(end_step, n_steps))
        
        current_sequence[start_step:end_step] = platform_current
            
        spikes = np.zeros(n_steps)
        voltages = np.zeros(n_steps)
        
        for i in range(n_steps):
            spike, voltage = self.step(current_sequence[i])
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
    # 计算刺激持续时间 - 通过检测非零电流的时间点
    current_nonzero_indices = np.where(current != 0)[0]
    if len(current_nonzero_indices) > 0:
        dt = time[1] - time[0] if len(time) > 1 else 1.0  # 推断时间步长
        stimulus_duration_sec = len(current_nonzero_indices) * dt / 1000.0  # 转换为秒
        firing_rate = spike_count / stimulus_duration_sec if stimulus_duration_sec > 0 else 0
    else:
        firing_rate = 0
    ax2.text(0.02, 0.98, f'Spike number: {spike_count}\n firing rate: {firing_rate:.2f} Hz', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_current_response(neuron_type, neuron, T, current_amplitudes, current_start, current_end):
    """
    分析神经元在不同电流强度下的响应
    
    参数:
    - neuron_type: 神经元类型
    - current_amplitudes: 电流强度列表
    - model_path: 模型文件路径
    - show_plot: 是否显示图像
    """
    
    # 创建多子图
    n_currents = len(current_amplitudes)
    fig, axes = plt.subplots(n_currents + 1, 1, figsize=(10, 2 * (n_currents + 1)), 
                            gridspec_kw={'hspace': 0.3})
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, n_currents))
    
    # 第一个子图：显示所有电流波形
    dt = neuron._dt  # 获取神经元的时间步长
    n_steps_ref = int(T / dt)
    time_ref = np.arange(n_steps_ref) * dt  # 参考时间序列，考虑dt
    for i, current_amp in enumerate(current_amplitudes):
        current_waveform = np.zeros(n_steps_ref)
        start_step = int(current_start / dt)
        end_step = int(current_end / dt)
        start_step = max(0, min(start_step, n_steps_ref))
        end_step = max(0, min(end_step, n_steps_ref))
        current_waveform[start_step:end_step] = current_amp
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
        # 计算firing rate：spike数 / 刺激持续时间(秒)
        stimulus_duration_sec = (current_end - current_start) / 1000.0  # 转换为秒
        firing_rate = spike_count / stimulus_duration_sec if stimulus_duration_sec > 0 else 0
        all_results.append({
            'current': current_amp,
            'spike_count': spike_count,
            'firing_rate': firing_rate,
            'voltage': voltage,
            'spikes': spikes
        })
        
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
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
import os
from matplotlib.patches import Rectangle

# 导入必要的模块
import sys
sys.path.append('..')
from models import BillehColumn

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def simulate_neuron_response(neuron_model_template_index, platform_current, 
                           model_path='../GLIF_network/network_dat.pkl',
                           T=1000, dt=1.0, current_start=200, current_end=800):
    """
    模拟单个神经元在平台电流刺激下的响应
    
    参数:
    - neuron_model_template_index: 目标神经元模板索引 (从0到111)
    - platform_current: 平台电流强度 (nA)
    - model_path: 模型文件路径
    - T: 总仿真时间步数 (ms)
    - dt: 时间步长 (ms)
    - current_start: 电流开始时间 (ms)
    - current_end: 电流结束时间 (ms)
    
    返回:
    - time: 时间序列
    - current: 输入电流序列
    - voltage: 膜电位序列
    - spikes: 脉冲序列 (0或1)
    """
    
    # 读取模型文件
    with open(model_path, 'rb') as f:
        d = pkl.load(f)
    
    def build_single_type_network(neuron_model_template_index, n_neurons=1):
        """构建单一类型神经元网络"""
        node_params = {k: np.array([v]) for k, v in d['nodes'][neuron_model_template_index]['params'].items()}
        
        # 添加虚拟突触连接
        indices = np.array([[0, 0]], dtype=np.int64)
        weights = np.array([1e-9], dtype=np.float32)  # 非零极小值
        delays = np.array([1.0], dtype=np.float32)
        
        return dict(
            node_params=node_params,
            node_type_ids=np.zeros(n_neurons, dtype=np.int64),
            n_nodes=n_neurons,
            synapses=dict(
                indices=indices,
                weights=weights,
                delays=delays,
                dense_shape=(4 * n_neurons, n_neurons)
            )
        )
    
    def build_dummy_input(n_neurons):
        """构建虚拟输入"""
        input_population = dict(
            indices=np.zeros((0, 2), dtype=np.int64),
            weights=np.zeros(0, dtype=np.float32),
            n_inputs=1
        )
        bkg_weights = np.zeros(n_neurons * 4, dtype=np.float32)
        return input_population, bkg_weights
    
    # 获取神经元类型索引
    print(f"正在模拟神经元类型: {neuron_model_template_index}, 电流强度: {platform_current} nA")
    
    # 构建网络和输入
    network = build_single_type_network(neuron_model_template_index)
    input_population, bkg_weights = build_dummy_input(1)
    
    # 创建神经元模型
    cell = BillehColumn(
        network, input_population, bkg_weights,
        dt=dt, train_recurrent=False, train_input=False, train_bkg=False
    )
    
    # 构建输入电流序列
    current_sequence = np.zeros(T, dtype=np.float32)
    current_sequence[current_start:current_end] = platform_current
    
    # 准备输入张量
    inputs = current_sequence.reshape(1, T, 1).astype(np.float32)
    
    # 初始化状态
    state = cell.zero_state(batch_size=1)
    
    # 运行仿真
    voltages = []
    spikes = []
    
    for t in range(T):
        input_t = tf.convert_to_tensor(inputs[:, t, :])
        (spike, voltage), state = cell(input_t, state)
        voltages.append(voltage.numpy().squeeze())
        spikes.append(spike.numpy().squeeze())
    
    # 转换为numpy数组
    time = np.arange(T) * dt
    voltages = np.array(voltages)
    spikes = np.array(spikes)
    
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
    ax1.plot(time, current, 'b-', linewidth=2, label=f'输入电流 ({current_amplitude} nA)')
    ax1.set_ylabel('电流 (nA)', fontsize=12)
    ax1.set_title(f'{neuron_type} 神经元响应 - 电流强度: {current_amplitude} nA', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 下方子图：膜电位和脉冲
    ax2.plot(time, voltage, 'k-', linewidth=1.5, label='膜电位')
    
    # 添加脉冲标记（在对应时间点的膜电位上画红色竖线）
    spike_indices = np.where(spikes > 0.5)[0]  # 找到脉冲发生的时间点索引
    for spike_idx in spike_indices:
        spike_voltage = voltage[spike_idx]  # 获取脉冲时刻的膜电位值
        ax2.plot(time[spike_idx], spike_voltage, '|k', markersize=10, color='red')
    
    ax2.set_xlabel('时间 (ms)', fontsize=12)
    ax2.set_ylabel('膜电位 (mV)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加脉冲计数信息
    spike_count = np.sum(spikes > 0.5)
    firing_rate = spike_count / (time[-1] / 1000)  # Hz
    ax2.text(0.02, 0.98, f'脉冲数: {spike_count}\n发放率: {firing_rate:.2f} Hz', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_current_response(neuron_type, current_amplitudes, 
                           model_path='../GLIF_network/network_dat.pkl'):
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
    time_ref = np.arange(1000)  # 参考时间序列
    for i, current_amp in enumerate(current_amplitudes):
        current_waveform = np.zeros(1000)
        current_waveform[200:800] = current_amp
        axes[0].plot(time_ref, current_waveform, color=colors[i], linewidth=2, 
                    label=f'{current_amp} nA')
    
    axes[0].set_ylabel('电流 (nA)', fontsize=12)
    axes[0].set_title(f'{neuron_type} 神经元在不同电流强度下的响应', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 存储结果用于分析
    all_results = []
    
    # 对每个电流强度进行仿真
    for i, current_amp in enumerate(current_amplitudes):
        print(f"\n处理电流强度: {current_amp} nA ({i+1}/{n_currents})")
        
        # 运行仿真
        time, current, voltage, spikes = simulate_neuron_response(
            neuron_type, current_amp, model_path=model_path
        )
        
        # 存储结果
        spike_count = np.sum(spikes > 0.5)
        firing_rate = spike_count / (time[-1] / 1000)
        all_results.append({
            'current': current_amp,
            'spike_count': spike_count,
            'firing_rate': firing_rate,
            'voltage': voltage,
            'spikes': spikes
        })
        
        # 绘制到对应子图
        ax = axes[i + 1]
        ax.plot(time, voltage, color='black', linewidth=1.5, label='膜电位')
        
        # 添加脉冲标记（在对应时间点的膜电位上画红色竖线）
        spike_indices = np.where(spikes > 0.5)[0]
        for spike_idx in spike_indices:
            spike_voltage = voltage[spike_idx]  # 获取脉冲时刻的膜电位值
            ax.plot(time[spike_idx], spike_voltage, '|k', markersize=10, color='red')
        
        ax.set_ylabel('膜电位 (mV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f'{current_amp} nA\n脉冲: {spike_count}\n频率: {firing_rate:.1f} Hz', 
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[-1].set_xlabel('时间 (ms)', fontsize=12)
    plt.tight_layout(pad=0.5)
    plt.show()
    
    # 绘制I-F曲线
    plot_if_curve(neuron_type, all_results)
    
    return all_results

def plot_if_curve(neuron_type, results):
    """
    绘制电流-发放频率(I-F)曲线
    
    参数:
    - neuron_type: 神经元类型
    - results: 仿真结果列表
    """
    
    currents = [r['current'] for r in results]
    firing_rates = [r['firing_rate'] for r in results]
    
    plt.figure(figsize=(5,3))
    plt.plot(currents, firing_rates, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('输入电流 (nA)', fontsize=12)
    plt.ylabel('发放频率 (Hz)', fontsize=12)
    plt.title(f'{neuron_type} 神经元的I-F曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加数据标签
    for i, (curr, rate) in enumerate(zip(currents, firing_rates)):
        plt.annotate(f'{rate:.1f}', (curr, rate), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
"""
交互式V1模型仿真测试工具
======================

该脚本提供了交互式的测试界面，允许用户：
1. 选择特定的神经元群体进行仿真
2. 自定义仿真参数
3. 实时查看结果
4. 导出特定神经元的详细数据

基于Training-data-driven-V1-model-test工具包
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict, Optional, Tuple

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_simulation import V1SimulationTester


class InteractiveV1Tester(V1SimulationTester):
    """
    交互式V1模型测试类
    
    继承自V1SimulationTester，添加了交互式功能
    """
    
    def select_neurons_by_criteria(self, network: Dict, 
                                 layer: Optional[str] = None,
                                 cell_type: Optional[str] = None,
                                 spatial_region: Optional[Tuple[float, float, float, float]] = None,
                                 neuron_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        根据指定条件选择神经元
        
        参数：
        network: 网络结构
        layer: 层名称 (如 'L2', 'L4', 'L5' 等)
        cell_type: 细胞类型 ('e'=兴奋性, 'i'=抑制性)
        spatial_region: 空间区域 (x_min, x_max, z_min, z_max) 单位：微米
        neuron_ids: 直接指定的神经元ID列表
        
        返回：
        selected_indices: 选中的神经元索引
        """
        # 初始化选择为所有神经元
        selected = np.ones(network['n_nodes'], dtype=bool)
        
        # 按层筛选
        if layer is not None and cell_type is not None:
            pop_name = f"{layer}{cell_type}"
            if pop_name in network['laminar_indices']:
                layer_indices = network['laminar_indices'][pop_name]
                layer_mask = np.zeros(network['n_nodes'], dtype=bool)
                layer_mask[layer_indices] = True
                selected = selected & layer_mask
                print(f"按层筛选 {pop_name}: {np.sum(layer_mask)} 个神经元")
        
        # 按空间位置筛选
        if spatial_region is not None:
            x_min, x_max, z_min, z_max = spatial_region
            spatial_mask = (
                (network['x'] >= x_min) & (network['x'] <= x_max) &
                (network['z'] >= z_min) & (network['z'] <= z_max)
            )
            selected = selected & spatial_mask
            print(f"按空间区域筛选: {np.sum(spatial_mask)} 个神经元")
        
        # 按ID筛选
        if neuron_ids is not None:
            id_mask = np.zeros(network['n_nodes'], dtype=bool)
            valid_ids = [nid for nid in neuron_ids if 0 <= nid < network['n_nodes']]
            id_mask[valid_ids] = True
            selected = selected & id_mask
            print(f"按ID筛选: {len(valid_ids)} 个神经元")
        
        selected_indices = np.where(selected)[0]
        print(f"最终选择: {len(selected_indices)} 个神经元")
        
        return selected_indices
    
    def analyze_selected_neurons(self, simulation_results: Dict, 
                               selected_indices: np.ndarray,
                               time_window: Optional[Tuple[float, float]] = None) -> Dict:
        """
        分析选定神经元的活动
        
        参数：
        simulation_results: 仿真结果
        selected_indices: 选定的神经元索引
        time_window: 时间窗口 (start_ms, end_ms)
        
        返回：
        analysis: 分析结果字典
        """
        # 提取选定神经元的数据
        spikes = simulation_results['spikes'][:, :, selected_indices]
        voltages = simulation_results['voltages'][:, :, selected_indices]
        
        # 时间窗口筛选
        if time_window is not None:
            start_idx = int(time_window[0] / self.dt)
            end_idx = int(time_window[1] / self.dt)
            spikes = spikes[:, start_idx:end_idx, :]
            voltages = voltages[:, start_idx:end_idx, :]
            time_axis = simulation_results['time_axis'][start_idx:end_idx]
        else:
            time_axis = simulation_results['time_axis']
        
        # 计算统计量
        # 1. 发放率（Hz）
        firing_rates = np.mean(spikes, axis=(0, 1)) * 1000 / self.dt
        
        # 2. 变异系数（CV）- 衡量发放规律性
        spike_counts = []
        bin_size = 50  # 50ms的时间窗
        n_bins = len(time_axis) // int(bin_size / self.dt)
        
        for n in range(spikes.shape[2]):
            neuron_spikes = spikes[0, :, n]
            binned_counts = []
            for b in range(n_bins):
                start = b * int(bin_size / self.dt)
                end = (b + 1) * int(bin_size / self.dt)
                count = np.sum(neuron_spikes[start:end])
                binned_counts.append(count)
            spike_counts.append(binned_counts)
        
        spike_counts = np.array(spike_counts)
        cv_values = np.std(spike_counts, axis=1) / (np.mean(spike_counts, axis=1) + 1e-10)
        
        # 3. 同步性指数 - 衡量群体同步程度
        population_activity = np.mean(spikes[0], axis=1)
        synchrony_index = np.std(population_activity) / np.mean(population_activity)
        
        # 4. 膜电位统计
        v_mean = np.mean(voltages)
        v_std = np.std(voltages)
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        analysis = {
            'n_neurons': len(selected_indices),
            'firing_rates': firing_rates,
            'mean_firing_rate': np.mean(firing_rates),
            'std_firing_rate': np.std(firing_rates),
            'cv_values': cv_values,
            'mean_cv': np.mean(cv_values),
            'synchrony_index': synchrony_index,
            'voltage_stats': {
                'mean': v_mean,
                'std': v_std,
                'min': v_min,
                'max': v_max
            },
            'selected_indices': selected_indices,
            'time_window': time_window,
            'processed_spikes': spikes,
            'processed_voltages': voltages,
            'time_axis': time_axis
        }
        
        return analysis
    
    def plot_detailed_activity(self, simulation_results: Dict,
                             selected_indices: np.ndarray,
                             analysis: Dict,
                             output_file: Optional[str] = None):
        """
        绘制详细的神经活动图
        
        参数：
        simulation_results: 仿真结果
        selected_indices: 选定的神经元索引
        analysis: 分析结果
        output_file: 输出文件路径
        """
        # 创建图形布局
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 光栅图（Raster plot）
        ax1 = fig.add_subplot(gs[0, :])
        # 使用分析结果中处理过的数据，确保维度匹配
        spikes = analysis['processed_spikes'][0, :, :]
        time_axis = analysis['time_axis']
        
        # 只绘制前50个神经元的光栅图
        n_show = min(50, len(selected_indices))
        for i in range(n_show):
            spike_times = time_axis[spikes[:, i] > 0]
            ax1.scatter(spike_times, np.ones_like(spike_times) * i, 
                       s=10, c='black', marker='|')
        
        ax1.set_xlabel('时间 (ms)')
        ax1.set_ylabel('神经元')
        ax1.set_title('脉冲光栅图')
        ax1.set_xlim([time_axis[0], min(time_axis[-1], time_axis[0] + 1000)])
        
        # 2. 群体发放率
        ax2 = fig.add_subplot(gs[1, :])
        pop_rate = np.mean(spikes, axis=1) * 1000 / self.dt
        
        # 使用滑动窗口平滑
        window_size = int(10 / self.dt)  # 10ms窗口
        smoothed_rate = np.convolve(pop_rate, np.ones(window_size)/window_size, mode='same')
        
        ax2.plot(time_axis, pop_rate, 'gray', alpha=0.5, label='瞬时')
        ax2.plot(time_axis, smoothed_rate, 'red', linewidth=2, label='平滑')
        ax2.set_xlabel('时间 (ms)')
        ax2.set_ylabel('发放率 (Hz)')
        ax2.set_title('群体平均发放率')
        ax2.legend()
        ax2.set_xlim([time_axis[0], min(time_axis[-1], time_axis[0] + 1000)])
        
        # 3. 发放率分布
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.hist(analysis['firing_rates'], bins=30, color='blue', alpha=0.7)
        ax3.axvline(analysis['mean_firing_rate'], color='red', linestyle='--', 
                   label=f'平均: {analysis["mean_firing_rate"]:.2f} Hz')
        ax3.set_xlabel('发放率 (Hz)')
        ax3.set_ylabel('神经元数量')
        ax3.set_title('发放率分布')
        ax3.legend()
        
        # 4. CV分布
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(analysis['cv_values'], bins=30, color='green', alpha=0.7)
        ax4.axvline(analysis['mean_cv'], color='red', linestyle='--',
                   label=f'平均CV: {analysis["mean_cv"]:.2f}')
        ax4.set_xlabel('变异系数 (CV)')
        ax4.set_ylabel('神经元数量')
        ax4.set_title('发放规律性')
        ax4.legend()
        
        # 5. 样本膜电位轨迹
        ax5 = fig.add_subplot(gs[2, 2])
        n_sample = min(5, len(selected_indices))
        # 使用分析结果中处理过的电压数据
        voltages = analysis['processed_voltages'][0, :, :n_sample]
        
        # 确保不超出时间轴范围
        max_time_points = min(500, len(time_axis))
        for i in range(n_sample):
            ax5.plot(time_axis[:max_time_points], voltages[:max_time_points, i], alpha=0.7)
        
        ax5.set_xlabel('时间 (ms)')
        ax5.set_ylabel('膜电位 (mV)')
        ax5.set_title('样本膜电位轨迹')
        ax5.set_xlim([time_axis[0], time_axis[max_time_points-1]])
        
        # 添加总体统计信息
        fig.suptitle(f'神经元群体活动分析 (n={analysis["n_neurons"]})', fontsize=16)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def export_neuron_data(self, simulation_results: Dict,
                         neuron_id: int,
                         output_file: str):
        """
        导出单个神经元的详细数据
        
        参数：
        simulation_results: 仿真结果
        neuron_id: 神经元ID
        output_file: 输出文件路径
        """
        # 提取该神经元的数据
        spikes = simulation_results['spikes'][0, :, neuron_id]
        voltage = simulation_results['voltages'][0, :, neuron_id]
        time_axis = simulation_results['time_axis']
        
        # 计算额外信息
        spike_times = time_axis[spikes > 0]
        
        # 如果有自适应电流数据
        if 'adaptive_currents' in simulation_results:
            asc = simulation_results['adaptive_currents'][0, :, neuron_id, :]
        else:
            asc = None
        
        # 保存数据
        data_dict = {
            'neuron_id': neuron_id,
            'time_axis': time_axis,
            'spikes': spikes,
            'voltage': voltage,
            'spike_times': spike_times,
            'firing_rate': len(spike_times) / (self.simulation_time / 1000),  # Hz
            'simulation_params': {
                'dt': self.dt,
                'duration': self.simulation_time
            }
        }
        
        if asc is not None:
            data_dict['adaptive_currents'] = asc
        
        # 根据文件扩展名选择保存格式
        if output_file.endswith('.npz'):
            np.savez_compressed(output_file, **data_dict)
        elif output_file.endswith('.csv'):
            import pandas as pd
            df = pd.DataFrame({
                'time_ms': time_axis,
                'spike': spikes,
                'voltage_mV': voltage
            })
            if asc is not None:
                df['asc_1'] = asc[:, 0]
                df['asc_2'] = asc[:, 1]
            df.to_csv(output_file, index=False)
        
        print(f"神经元 {neuron_id} 的数据已保存到: {output_file}")
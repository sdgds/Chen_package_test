"""
V1模型仿真测试工具
==================

该脚本用于测试Training-data-driven-V1-model-test工具包的仿真功能。
主要功能包括：
1. 导入Allen V1模型的网络结构和输入数据
2. 使用BillehColumn类进行神经网络仿真
3. 输出神经元的发放脉冲信号(spike)和膜电位(membrane potential)
4. 支持选择特定层、细胞类型的神经元进行分析
"""

import os
import sys
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import argparse
warnings.filterwarnings('ignore')

# 将上级目录添加到Python路径，以便导入工具包模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具包中的关键模块
from load_sparse import load_network, load_input, set_laminar_indices
from models import BillehColumn
from classification_tools import create_model
import pandas as pd


class V1SimulationTester:
    """
    V1模型仿真测试类
    
    该类封装了V1模型的仿真测试功能，包括：
    - 加载网络结构和输入数据
    - 运行神经网络仿真
    - 提取和分析神经元活动
    """
    
    def __init__(self, data_dir='../GLIF_network', 
                 simulation_time=1000,  # 仿真时长（毫秒）
                 dt=1.0,               # 时间步长（毫秒）
                 seed=42):
        """
        初始化仿真测试器
        
        参数：
        data_dir: GLIF_network数据目录路径
        simulation_time: 仿真时长（毫秒）
        dt: 时间步长（毫秒）
        seed: 随机种子
        """
        self.data_dir = data_dir
        self.simulation_time = simulation_time
        self.dt = dt
        self.seed = seed
        
        # 设置随机种子以确保可重复性
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        print(f"初始化V1模型仿真测试器...")
        print(f"数据目录: {data_dir}")
        print(f"仿真时长: {simulation_time} ms")
        print(f"时间步长: {dt} ms")
        
    def load_network_and_input(self, n_neurons=None, core_only=True):
        """
        加载网络结构和输入数据
        
        参数：
        n_neurons: 要使用的神经元数量（None表示使用所有核心神经元）
        core_only: 是否只使用核心区域的神经元（半径<400μm）
        
        返回：
        network: 网络结构字典
        input_populations: 输入信号列表（[LGN输入, 背景输入]）
        """
        print("\n加载网络结构...")
        
        # 加载网络结构
        # network字典包含：
        # - n_nodes: 神经元数量
        # - node_params: 神经元参数（V_th阈值电位、g电导、E_L静息电位等）
        # - node_type_ids: 每个神经元的类型ID
        # - synapses: 突触连接信息（indices、weights、delays）
        # - x,y,z: 神经元的空间坐标
        network = load_network(
            path=os.path.join(self.data_dir, 'network_dat.pkl'),
            h5_path=os.path.join(self.data_dir, 'network/v1_nodes.h5'),
            data_dir=self.data_dir,
            core_only=core_only,
            n_neurons=n_neurons,
            seed=self.seed
        )
        
        print(f"网络加载完成：{network['n_nodes']}个神经元，{network['n_edges']}个突触连接")
        
        # 加载输入数据
        print("\n加载输入数据...")
        # input_populations包含两个输入源：
        # [0]: LGN（外侧膝状体）输入 - 视觉刺激信号
        # [1]: Background（背景）输入 - 大脑其他区域的输入
        input_populations = load_input(
            path=os.path.join(self.data_dir, 'input_dat.pkl'),
            start=0,
            duration=self.simulation_time,
            dt=self.dt,
            bmtk_id_to_tf_id=network['bmtk_id_to_tf_id']
        )
        
        lgn_input = input_populations[0]
        bkg_input = input_populations[1]
        
        print(f"LGN输入：{lgn_input['n_inputs']}个输入神经元")
        print(f"背景输入：{bkg_input['n_inputs']}个输入神经元")
        
        # 设置层级索引（用于后续按层分析）
        df_node_types = pd.read_csv(
            os.path.join(self.data_dir, 'network/v1_node_types.csv'), 
            delimiter=' '
        )
        network = set_laminar_indices(df_node_types, 
                                    os.path.join(self.data_dir, 'network/v1_nodes.h5'), 
                                    network)
        
        return network, input_populations
    
    def prepare_simulation(self, network, input_populations):
        """
        准备仿真所需的参数和数据
        
        参数：
        network: 网络结构
        input_populations: 输入信号
        
        返回：
        cell: BillehColumn神经元模型
        input_spikes: 输入脉冲信号
        bkg_weights: 背景输入权重
        """
        # 提取LGN和背景输入
        lgn_input = input_populations[0]
        bkg_input = input_populations[1]
        
        # 计算背景输入权重
        # 背景输入代表来自大脑其他区域的非特异性输入
        bkg_weights = np.zeros(network['n_nodes'] * 4)  # 4种受体类型
        for idx, weight in zip(bkg_input['indices'], bkg_input['weights']):
            bkg_weights[idx[0]] += weight
        
        print(f"\n背景输入权重统计：")
        print(f"  平均值: {np.mean(bkg_weights):.4f}")
        print(f"  标准差: {np.std(bkg_weights):.4f}")
        print(f"  最小值: {np.min(bkg_weights):.4f}")
        print(f"  最大值: {np.max(bkg_weights):.4f}")
        
        # 创建BillehColumn模型
        # 该模型实现了GLIF3（广义漏积分发放）神经元动力学
        print("\n创建BillehColumn神经元模型...")
        cell = BillehColumn(
            network=network,
            input_population=lgn_input,
            bkg_weights=bkg_weights,
            dt=self.dt,
            gauss_std=0.5,          # 脉冲梯度的高斯标准差
            dampening_factor=0.3,    # 脉冲梯度的阻尼因子
            train_recurrent=False,   # 不训练递归权重（仅测试）
            train_input=False,       # 不训练输入权重（仅测试）
            _return_interal_variables=True  # 返回内部变量用于分析
        )
        
        return cell, lgn_input, bkg_weights
    
    def run_simulation(self, cell, lgn_input, batch_size=1, use_rnn_layer=False):
        """
        运行神经网络仿真
        
        参数：
        cell: BillehColumn神经元模型
        lgn_input: LGN输入数据
        batch_size: 批次大小
        use_rnn_layer: 是否使用TensorFlow RNN层（True=原始方法，False=逐时间步方法）
        
        返回：
        simulation_results: 包含脉冲、膜电位等的仿真结果字典
        """
        method_name = "TensorFlow RNN层" if use_rnn_layer else "逐时间步循环"
        print(f"\n开始运行仿真（方法：{method_name}，批次大小={batch_size}）...")
        
        # 准备输入数据
        n_timesteps = int(self.simulation_time / self.dt)
        input_spikes = lgn_input['spikes']
        input_spikes = tf.convert_to_tensor(input_spikes, dtype=tf.float32)
        input_spikes = tf.expand_dims(input_spikes, 0)  # 添加批次维度
        input_spikes = tf.tile(input_spikes, [batch_size, 1, 1])
        
        if use_rnn_layer:
            # 方法1：使用TensorFlow RNN层（原始工具包方法）
            return self._run_rnn_simulation(cell, input_spikes, batch_size, n_timesteps)
        else:
            # 方法2：逐时间步手动循环（测试工具包方法）
            return self._run_manual_simulation(cell, input_spikes, batch_size, n_timesteps)
    
    def _run_rnn_simulation(self, cell, input_spikes, batch_size, n_timesteps):
        """
        使用TensorFlow RNN层的仿真方法（原始工具包方法）
        
        优点：
        - 计算效率高，GPU并行优化
        - 内存管理优化
        - 支持自动微分和训练
        
        适用场景：
        - 大规模仿真
        - 模型训练
        - 生产环境使用
        """
        print("使用TensorFlow RNN层方法...")
        import time
        start_time = time.time()
        
        from models import SparseLayer
        
        # 构建输入层
        input_layer = SparseLayer(
            indices=cell.input_indices,
            weights=cell.input_weight_values,
            dense_shape=cell.input_dense_shape,
            bkg_weights=cell.bkg_weights,
            use_decoded_noise=False,
            dtype=tf.float32
        )
        
        # 处理输入
        rnn_inputs = input_layer(input_spikes)
        
        # 初始状态
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        
        # 创建RNN层
        rnn = tf.keras.layers.RNN(
            cell, 
            return_sequences=True, 
            return_state=False,
            name='billeh_rnn'
        )
        
        # 运行RNN - 一次性处理所有时间步
        outputs = rnn(rnn_inputs, initial_state=initial_state)
        
        # 提取结果
        all_spikes = outputs[0]       # (batch, time, neurons)
        all_voltages = outputs[1]     # (batch, time, neurons)
        all_ascs = outputs[2]         # (batch, time, neurons, 2)
        all_psc_rise = outputs[3]     # (batch, time, neurons*4)
        all_psc = outputs[4]          # (batch, time, neurons*4)
        
        computation_time = time.time() - start_time
        print(f"RNN层仿真完成！耗时: {computation_time:.3f} 秒")
        
        # 计算统计信息
        spike_rates = tf.reduce_mean(all_spikes, axis=(0, 1)).numpy()
        mean_rate = np.mean(spike_rates)
        
        print(f"\n仿真统计：")
        print(f"  平均发放率: {mean_rate:.2f} Hz")
        print(f"  活跃神经元比例: {np.mean(spike_rates > 0.1):.2%}")
        
        simulation_results = {
            'spikes': all_spikes.numpy(),
            'voltages': all_voltages.numpy(),
            'adaptive_currents': all_ascs.numpy(),
            'psc_rise': all_psc_rise.numpy(),
            'psc': all_psc.numpy(),
            'spike_rates': spike_rates,
            'time_axis': np.arange(n_timesteps) * self.dt,
            'computation_time': computation_time,
            'method': 'RNN层'
        }
        
        return simulation_results
    
    def _run_manual_simulation(self, cell, input_spikes, batch_size, n_timesteps):
        """
        逐时间步手动循环的仿真方法（测试工具包方法）
        
        优点：
        - 容易理解和调试
        - 可以在每个时间步插入自定义逻辑
        - 便于监控和分析
        
        缺点：
        - 计算效率较低
        - Python循环开销
        
        适用场景：
        - 算法调试
        - 详细分析
        - 自定义监控
        """
        print("使用逐时间步循环方法...")
        import time
        start_time = time.time()
        
        from models import SparseLayer
        
        # 初始化神经元状态
        # 状态包括：
        # - z_buffer: 脉冲缓冲区（用于实现突触延迟）
        # - v: 膜电位
        # - r: 不应期
        # - asc_1, asc_2: 两个自适应电流
        # - psc_rise, psc: 突触后电流（上升相和衰减相）
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        
        # 通过输入层转换输入脉冲为输入电流
        input_layer = SparseLayer(
            indices=cell.input_indices,
            weights=cell.input_weight_values,
            dense_shape=cell.input_dense_shape,
            bkg_weights=cell.bkg_weights,
            use_decoded_noise=False,
            dtype=tf.float32
        )
        
        # 计算输入电流
        input_currents = input_layer(input_spikes)
        
        # 运行仿真
        print("运行神经动力学仿真...")
        
        # 存储结果
        spikes_list = []
        voltages_list = []
        asc_list = []
        psc_rise_list = []
        psc_list = []
        
        state = initial_state
        
        # 逐时间步仿真 - 手动循环
        for t in range(n_timesteps):
            if t % 100 == 0:
                print(f"  仿真进度: {t}/{n_timesteps} 时间步")
            
            # 获取当前时间步的输入
            current_input = input_currents[:, t, :]
            
            # 运行一个时间步的神经动力学
            # 这里调用BillehColumn.call()方法，实现了：
            # 1. GLIF3神经元膜电位更新
            # 2. 突触后电流计算
            # 3. 自适应电流更新
            # 4. 脉冲生成和不应期处理
            outputs, state = cell(current_input, state)
            
            # 提取输出
            # outputs包含: (spikes, voltages, ascs, psc_rise, psc)
            spikes = outputs[0]    # 脉冲发放（0或1）
            voltages = outputs[1]  # 膜电位（mV）
            ascs = outputs[2]      # 自适应电流（两个分量）
            psc_rise = outputs[3]  # 突触后电流上升相
            psc = outputs[4]       # 突触后电流
            
            spikes_list.append(spikes)
            voltages_list.append(voltages)
            asc_list.append(ascs)
            psc_rise_list.append(psc_rise)
            psc_list.append(psc)
        
        # 堆叠结果
        all_spikes = tf.stack(spikes_list, axis=1)      # (batch, time, neurons)
        all_voltages = tf.stack(voltages_list, axis=1)  # (batch, time, neurons)
        all_ascs = tf.stack(asc_list, axis=1)          # (batch, time, neurons, 2)
        all_psc_rise = tf.stack(psc_rise_list, axis=1) # (batch, time, neurons*4)
        all_psc = tf.stack(psc_list, axis=1)           # (batch, time, neurons*4)
        
        computation_time = time.time() - start_time
        print(f"逐时间步仿真完成！耗时: {computation_time:.3f} 秒")
        
        # 计算统计信息
        spike_rates = tf.reduce_mean(all_spikes, axis=(0, 1)).numpy()
        mean_rate = np.mean(spike_rates)
        
        print(f"\n仿真统计：")
        print(f"  平均发放率: {mean_rate:.2f} Hz")
        print(f"  活跃神经元比例: {np.mean(spike_rates > 0.1):.2%}")
        
        simulation_results = {
            'spikes': all_spikes.numpy(),
            'voltages': all_voltages.numpy(),
            'adaptive_currents': all_ascs.numpy(),
            'psc_rise': all_psc_rise.numpy(),
            'psc': all_psc.numpy(),
            'spike_rates': spike_rates,
            'time_axis': np.arange(n_timesteps) * self.dt,
            'computation_time': computation_time,
            'method': '逐时间步'
        }
        
        return simulation_results
    
    def analyze_by_layer_and_type(self, network, simulation_results):
        """
        按层和细胞类型分析仿真结果
        
        参数：
        network: 网络结构
        simulation_results: 仿真结果
        
        返回：
        analysis_results: 分析结果字典
        """
        print("\n按层和细胞类型分析神经活动...")
        
        analysis_results = {}
        
        # 定义要分析的层
        layers = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        cell_types = ['e', 'i']  # 兴奋性和抑制性
        
        for layer in layers:
            for cell_type in cell_types:
                pop_name = f"{layer}{cell_type}"
                
                # 检查该群体是否存在
                if pop_name not in network['laminar_indices']:
                    continue
                
                # 获取该群体的神经元索引
                neuron_indices = network['laminar_indices'][pop_name]
                
                if len(neuron_indices) == 0:
                    continue
                
                # 提取该群体的活动
                pop_spikes = simulation_results['spikes'][:, :, neuron_indices]
                pop_voltages = simulation_results['voltages'][:, :, neuron_indices]
                
                # 计算统计信息
                mean_rate = np.mean(pop_spikes) * 1000 / self.dt  # 转换为Hz
                
                # 计算膜电位统计
                mean_voltage = np.mean(pop_voltages)
                std_voltage = np.std(pop_voltages)
                
                analysis_results[pop_name] = {
                    'neuron_count': len(neuron_indices),
                    'mean_firing_rate': mean_rate,
                    'mean_voltage': mean_voltage,
                    'std_voltage': std_voltage,
                    'neuron_indices': neuron_indices
                }
                
                cell_type_name = "兴奋性" if cell_type == 'e' else "抑制性"
                print(f"  {layer}层{cell_type_name}神经元 ({pop_name}):")
                print(f"    神经元数量: {len(neuron_indices)}")
                print(f"    平均发放率: {mean_rate:.2f} Hz")
                print(f"    平均膜电位: {mean_voltage:.2f} mV")
                
        return analysis_results
    
    def save_results(self, simulation_results, analysis_results, output_dir='results'):
        """
        保存仿真结果
        
        参数：
        simulation_results: 仿真结果
        analysis_results: 分析结果
        output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始数据
        np.savez_compressed(
            os.path.join(output_dir, f'simulation_results_{timestamp}.npz'),
            **simulation_results
        )
        
        # 保存分析结果
        with open(os.path.join(output_dir, f'analysis_results_{timestamp}.pkl'), 'wb') as f:
            pkl.dump(analysis_results, f)
        
        print(f"\n结果已保存到: {output_dir}")
    
    def plot_sample_activity(self, network, simulation_results, analysis_results, 
                            sample_neurons=10, output_dir='figures'):
        """
        绘制样本神经元活动
        
        参数：
        network: 网络结构
        simulation_results: 仿真结果
        analysis_results: 分析结果
        sample_neurons: 每种类型要绘制的神经元数量
        output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个层级和类型绘制样本神经元
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for layer in ['L2', 'L4', 'L5']:
            for cell_type in ['e', 'i']:
                pop_name = f"{layer}{cell_type}"
                
                if pop_name not in analysis_results:
                    continue
                
                # 获取神经元索引
                neuron_indices = analysis_results[pop_name]['neuron_indices']
                
                if len(neuron_indices) < sample_neurons:
                    sample_indices = neuron_indices
                else:
                    sample_indices = np.random.choice(neuron_indices, sample_neurons, replace=False)
                
                # 绘制膜电位
                ax = axes[plot_idx]
                time_axis = simulation_results['time_axis']
                
                for i, neuron_idx in enumerate(sample_indices):
                    voltage = simulation_results['voltages'][0, :, neuron_idx]
                    ax.plot(time_axis, voltage + i * 20, 'k', linewidth=0.5)
                    
                    # 标记脉冲
                    spike_times = time_axis[simulation_results['spikes'][0, :, neuron_idx] > 0]
                    for spike_time in spike_times:
                        ax.plot([spike_time, spike_time], 
                               [i * 20 - 5, i * 20 + 5], 'r', linewidth=1)
                
                cell_type_name = "Exc" if cell_type == 'e' else "Inh"
                ax.set_title(f"{layer} layer {cell_type_name} neuron")
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("voltage (mV)")
                ax.set_xlim([0, min(1000, self.simulation_time)])
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_neuron_activity.png'), dpi=300)
        plt.close()
        
        print(f"\n活动图已保存到: {output_dir}")


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='V1模型仿真测试工具 - 支持指定数据文件夹和输出文件夹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python test_simulation.py GLIF_network simulation_results
  python test_simulation.py ../GLIF_network ./results --simulation-time 2000 --dt 0.5
  python test_simulation.py Converted_param output_folder --n-neurons 1000 --core-only
        """
    )
    
    parser.add_argument(
        'data_dir',
        help='数据文件夹路径（包含network_dat.pkl、input_dat.pkl和network子文件夹）'
    )
    
    parser.add_argument(
        'output_dir',
        help='输出文件夹路径（保存仿真结果和分析结果）'
    )
    
    parser.add_argument(
        '--simulation-time',
        type=int,
        default=1000,
        help='仿真时长（毫秒，默认：1000）'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=1.0,
        help='时间步长（毫秒，默认：1.0）'
    )
    
    parser.add_argument(
        '--n-neurons',
        type=int,
        default=None,
        help='使用的神经元数量（默认：使用所有神经元）'
    )
    
    parser.add_argument(
        '--core-only',
        action='store_true',
        help='是否只使用核心区域神经元（半径<400μm）'
    )
    
    parser.add_argument(
        '--use-rnn-layer',
        action='store_true',
        help='使用TensorFlow RNN层方法（默认：逐时间步方法）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='批次大小（默认：1）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认：42）'
    )
    
    parser.add_argument(
        '--plot-activity',
        action='store_true',
        help='绘制样本神经元活动图'
    )
    
    return parser.parse_args()


def main():
    """
    主测试函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 60)
    print("V1模型仿真测试")
    print("=" * 60)
    print(f"数据文件夹: {args.data_dir}")
    print(f"输出文件夹: {args.output_dir}")
    print(f"仿真时长: {args.simulation_time} ms")
    print(f"时间步长: {args.dt} ms")
    print(f"神经元数量: {args.n_neurons if args.n_neurons else '全部'}")
    print(f"仅核心区域: {'是' if args.core_only else '否'}")
    print(f"仿真方法: {'RNN层' if args.use_rnn_layer else '逐时间步'}")
    
    # 创建测试器
    tester = V1SimulationTester(
        data_dir=args.data_dir,
        simulation_time=args.simulation_time,
        dt=args.dt,
        seed=args.seed
    )
    
    # 加载网络和输入
    network, input_populations = tester.load_network_and_input(
        n_neurons=args.n_neurons,
        core_only=args.core_only
    )
    
    # 准备仿真
    cell, lgn_input, bkg_weights = tester.prepare_simulation(network, input_populations)
    
    # 运行仿真
    method_name = "RNN层方法：原始工具包方法，计算更快" if args.use_rnn_layer else "逐时间步方法：便于调试和理解，计算较慢"
    print(f"\n使用{method_name}")
    simulation_results = tester.run_simulation(
        cell, lgn_input, 
        batch_size=args.batch_size, 
        use_rnn_layer=args.use_rnn_layer
    )
    
    # 分析结果
    analysis_results = tester.analyze_by_layer_and_type(network, simulation_results)
    
    # 保存结果
    tester.save_results(simulation_results, analysis_results, output_dir=args.output_dir)
    
    # 绘制样本活动（可选）
    if args.plot_activity:
        figures_dir = os.path.join(args.output_dir, 'figures')
        tester.plot_sample_activity(network, simulation_results, analysis_results, output_dir=figures_dir)
    
    print("\n测试完成！")
    print(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
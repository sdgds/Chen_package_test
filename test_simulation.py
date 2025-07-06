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
import h5py
warnings.filterwarnings('ignore')

# 将上级目录添加到Python路径，以便导入工具包模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具包中的关键模块
from load_sparse import load_network, load_input, set_laminar_indices
from models import BillehColumn
from classification_tools import create_model
import pandas as pd


class SparseLayerWithExternalBkg(tf.keras.layers.Layer):
    """
    修改版的SparseLayer，支持外部背景输入spikes
    
    与原始SparseLayer的区别：
    - 接受外部bkg_input spikes而不是内部生成噪声
    - 分别处理lgn_input和bkg_input，然后合并
    """
    def __init__(self, lgn_indices, lgn_weights, lgn_dense_shape, 
                 bkg_indices, bkg_weights, bkg_dense_shape, dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        # LGN输入参数
        self._lgn_indices = lgn_indices
        self._lgn_weights = tf.cast(lgn_weights, dtype)
        self._lgn_dense_shape = lgn_dense_shape
        
        # 背景输入参数 - 确保数据类型一致性
        self._bkg_indices = bkg_indices
        self._bkg_weights = tf.cast(bkg_weights, dtype)
        self._bkg_dense_shape = bkg_dense_shape
        
        self._dtype = dtype
        self._compute_dtype = dtype

    def call(self, inputs):
        """
        处理LGN和背景输入
        
        参数:
        inputs: 包含两个元素的tuple/list
                inputs[0]: lgn_spikes (batch, time, lgn_neurons)
                inputs[1]: bkg_spikes (batch, time, bkg_neurons)
        
        返回:
        input_current: 合并后的输入电流 (batch, time, neurons*4)
        """
        lgn_spikes, bkg_spikes = inputs
        
        # 获取输入形状
        lgn_tf_shp = tf.unstack(tf.shape(lgn_spikes))
        lgn_shp = lgn_spikes.shape.as_list()
        for i, a in enumerate(lgn_shp):
            if a is None:
                lgn_shp[i] = lgn_tf_shp[i]
        
        bkg_tf_shp = tf.unstack(tf.shape(bkg_spikes))
        bkg_shp = bkg_spikes.shape.as_list()
        for i, a in enumerate(bkg_shp):
            if a is None:
                bkg_shp[i] = bkg_tf_shp[i]
        
        # 处理LGN输入 - 参照原始SparseLayer的lgn处理方式
        lgn_sparse_w_in = tf.sparse.SparseTensor(
            self._lgn_indices, self._lgn_weights, self._lgn_dense_shape)
        lgn_inp = tf.reshape(lgn_spikes, (lgn_shp[0] * lgn_shp[1], lgn_shp[2]))
        
        lgn_current = tf.sparse.sparse_dense_matmul(
            lgn_sparse_w_in, tf.cast(lgn_inp, tf.float32), adjoint_b=True)
        lgn_current = tf.transpose(lgn_current)
        lgn_current = tf.cast(lgn_current, self._dtype)
        lgn_current = tf.reshape(lgn_current, (lgn_shp[0], lgn_shp[1], -1))
        
        # 处理背景输入 - 使用相同的稀疏矩阵乘法方式
        bkg_sparse_w_in = tf.sparse.SparseTensor(
            self._bkg_indices, self._bkg_weights, self._bkg_dense_shape)
        bkg_inp = tf.reshape(bkg_spikes, (bkg_shp[0] * bkg_shp[1], bkg_shp[2]))
        
        bkg_current = tf.sparse.sparse_dense_matmul(
            bkg_sparse_w_in, tf.cast(bkg_inp, tf.float32), adjoint_b=True)
        bkg_current = tf.transpose(bkg_current)
        bkg_current = tf.cast(bkg_current, self._dtype)
        bkg_current = tf.reshape(bkg_current, (bkg_shp[0], bkg_shp[1], -1))
        
        # 合并LGN和背景输入电流
        total_current = lgn_current + bkg_current
        
        return total_current
    

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
        准备使用外部背景输入的仿真参数
        
        参数：
        network: 网络结构
        input_populations: 输入信号，包含lgn_input和bkg_input
        
        返回：
        cell: BillehColumn神经元模型（背景权重设为0）
        lgn_input: LGN输入数据
        bkg_input: 背景输入数据
        """
        # 提取LGN和背景输入
        lgn_input = input_populations[0]
        bkg_input = input_populations[1]
        
        # 将背景权重设为0，因为我们要使用外部背景输入
        bkg_weights = np.zeros(network['n_nodes'] * 4)  # 4种受体类型
        
        # 创建BillehColumn模型
        print("\n创建BillehColumn神经元模型（外部背景输入模式）...")
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
        
        return cell, lgn_input, bkg_input
    
    def run_simulation(self, cell, lgn_input, bkg_input, batch_size=1):
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
        
        # 准备输入数据
        n_timesteps = int(self.simulation_time / self.dt)
        
        # LGN输入spikes
        lgn_spikes = lgn_input['spikes']
        lgn_spikes = tf.convert_to_tensor(lgn_spikes, dtype=tf.float32)
        lgn_spikes = tf.expand_dims(lgn_spikes, 0)  # 添加批次维度
        lgn_spikes = tf.tile(lgn_spikes, [batch_size, 1, 1])
        
        # 背景输入spikes
        bkg_spikes = bkg_input['spikes']
        bkg_spikes = tf.convert_to_tensor(bkg_spikes, dtype=tf.float32)
        bkg_spikes = tf.expand_dims(bkg_spikes, 0)  # 添加批次维度
        bkg_spikes = tf.tile(bkg_spikes, [batch_size, 1, 1])
        
        return self._run_manual_simulation(
               cell, lgn_spikes, bkg_spikes, lgn_input, bkg_input, batch_size, n_timesteps)

    def _run_manual_simulation(self, cell, lgn_spikes, bkg_spikes, 
                               lgn_input, bkg_input, batch_size, n_timesteps):
        """
        逐时间步手动循环的仿真方法（测试工具包方法）
        """

        import time
        start_time = time.time()
        
        # 初始化神经元状态
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        
        # 创建支持外部背景输入的输入层
        input_layer = SparseLayerWithExternalBkg(
            lgn_indices=cell.input_indices,
            lgn_weights=cell.input_weight_values,
            lgn_dense_shape=cell.input_dense_shape,
            bkg_indices=bkg_input['indices'],
            bkg_weights=bkg_input['weights'],
            bkg_dense_shape=(cell._n_receptors * cell._n_neurons, bkg_input['n_inputs']),
            dtype=tf.float32
        )
        
        # 计算输入电流
        input_currents = input_layer([lgn_spikes, bkg_spikes])
        
        # 运行仿真
        print("运行神经动力学仿真...")
        
        # 存储结果
        spikes_list = []
        voltages_list = []
        asc_list = []
        psc_rise_list = []
        psc_list = []
        
        state = initial_state
        
        # 逐时间步仿真
        for t in range(n_timesteps):
            if t % 100 == 0:
                print(f"  仿真进度: {t}/{n_timesteps} 时间步")
            
            # 获取当前时间步的输入
            current_input = input_currents[:, t, :]
            
            # 运行一个时间步的神经动力学
            outputs, state = cell(current_input, state)
            
            # 提取输出
            spikes = outputs[0]
            voltages = outputs[1]
            ascs = outputs[2]
            psc_rise = outputs[3]
            psc = outputs[4]
            
            spikes_list.append(spikes)
            voltages_list.append(voltages)
            asc_list.append(ascs)
            psc_rise_list.append(psc_rise)
            psc_list.append(psc)
        
        # 堆叠结果
        all_spikes = tf.stack(spikes_list, axis=1)
        all_voltages = tf.stack(voltages_list, axis=1)
        all_ascs = tf.stack(asc_list, axis=1)
        all_psc_rise = tf.stack(psc_rise_list, axis=1)
        all_psc = tf.stack(psc_list, axis=1)
        
        computation_time = time.time() - start_time
        print(f"仿真完成！耗时: {computation_time:.3f} 秒")
        
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
            'method': '外部背景输入'
        }
        
        return simulation_results
    
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
    
    def save_spikes_to_h5(self, simulation_results, network, output_file, 
                         selected_indices=None, metadata=None):
        """
        将仿真结果保存为与spikes_resting_3s_1500fr.h5相同格式的HDF5文件
        
        参数：
        simulation_results: 仿真结果字典，包含'spikes'和'time_axis'
        network: 网络结构字典，包含神经元ID映射信息
        output_file: 输出HDF5文件路径
        selected_indices: 可选，选择特定神经元的索引
        metadata: 可选，额外的元数据字典
        
        HDF5文件结构：
        /spikes/v1/timestamps - 脉冲时间戳 (ms)
        /spikes/v1/node_ids - 神经元节点ID
        """
        print(f"\n将仿真结果保存为HDF5格式: {output_file}")
        
        # 提取数据 (batch=0)
        spikes = simulation_results['spikes'][0]  # shape: (time, neurons)
        time_axis = simulation_results['time_axis']
        
        # 如果指定了特定神经元，则只保存这些神经元的数据
        if selected_indices is not None:
            spikes = spikes[:, selected_indices]
            print(f"选择了 {len(selected_indices)} 个神经元")
        
        # 转换为timestamps和node_ids格式
        timestamps = []
        node_ids = []
        
        print("转换脉冲数据为timestamps和node_ids格式...")
        
        for neuron_idx in range(spikes.shape[1]):
            # 找到该神经元的脉冲时间点
            spike_indices = np.where(spikes[:, neuron_idx] > 0)[0]
            
            if len(spike_indices) > 0:
                # 直接使用离散时间步，不进行精确时间估算
                spike_times = time_axis[spike_indices]
                timestamps.extend(spike_times)
                
                # 确定真实的神经元ID
                if selected_indices is not None:
                    real_neuron_id = selected_indices[neuron_idx]
                else:
                    real_neuron_id = neuron_idx
                
                # 如果网络中有bmtk_id映射，使用真实的BMTK ID
                if 'tf_id_to_bmtk_id' in network:
                    tf_to_bmtk_mapping = network['tf_id_to_bmtk_id']
                    # 检查映射是否为字典类型
                    if isinstance(tf_to_bmtk_mapping, dict):
                        bmtk_id = tf_to_bmtk_mapping.get(real_neuron_id, real_neuron_id)
                    elif isinstance(tf_to_bmtk_mapping, np.ndarray):
                        # 如果是numpy数组，使用索引访问
                        if real_neuron_id < len(tf_to_bmtk_mapping):
                            bmtk_id = tf_to_bmtk_mapping[real_neuron_id]
                        else:
                            bmtk_id = real_neuron_id
                    else:
                        # 其他类型，直接使用原始ID
                        bmtk_id = real_neuron_id
                else:
                    bmtk_id = real_neuron_id
                
                node_ids.extend([bmtk_id] * len(spike_times))
        
        timestamps = np.array(timestamps, dtype=np.float64)
        node_ids = np.array(node_ids, dtype=np.int64)
        
        print(f"总共 {len(timestamps)} 个脉冲事件")
        print(f"时间范围: {np.min(timestamps):.2f} - {np.max(timestamps):.2f} ms")
        print(f"涉及神经元: {len(np.unique(node_ids))} 个")
        print(f"神经元ID范围: {np.min(node_ids)} - {np.max(node_ids)}")
        
        # 保存为HDF5文件
        with h5py.File(output_file, 'w') as f:
            # 创建spikes组
            spikes_group = f.create_group('spikes')
            
            # 创建v1子组
            v1_group = spikes_group.create_group('v1')
            
            # 保存timestamps数据集
            timestamps_dataset = v1_group.create_dataset(
                'timestamps', 
                data=timestamps,
                dtype=np.float64,
                compression='gzip',
                compression_opts=9
            )
            timestamps_dataset.attrs['units'] = b'ms'
            
            # 保存node_ids数据集
            node_ids_dataset = v1_group.create_dataset(
                'node_ids',
                data=node_ids, 
                dtype=np.int64,
                compression='gzip',
                compression_opts=9
            )
            
            # 添加v1组的属性
            v1_group.attrs['sorting'] = 'none'
            
            # 添加仿真元数据
            if metadata is None:
                metadata = {}
            
            # 默认元数据
            default_metadata = {
                'simulation_time_ms': self.simulation_time,
                'dt_ms': self.dt,
                'n_neurons_simulated': spikes.shape[1],
                'n_spikes_total': len(timestamps),
                'creation_time': datetime.now().isoformat(),
                'source': 'V1SimulationTester'
            }
            
            # 合并元数据
            all_metadata = {**default_metadata, **metadata}
            
            # 将元数据添加到根组
            for key, value in all_metadata.items():
                f.attrs[key] = value
        
        print(f"HDF5文件保存完成: {output_file}")
        
        return output_file
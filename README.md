# Chen Package Test - V1模型仿真测试工具包

## 概述

Chen Package Test是一个专门用于测试和验证[Training-data-driven-V1-model](https://github.com/ifgovh/Training-data-driven-V1-model-test)工具包的仿真测试框架。该工具包基于Allen研究所的小鼠V1（初级视觉皮层）模型，实现了GLIF3（广义漏积分发放）神经元模型的大规模网络仿真。

## 数据准备

确保您有以下格式的数据文件：

```
Training-data-driven-V1-model
├── Chen工具包的所有内容
    Chen_package_test (我们的测试工具包放在Chen工具包文件夹下)
    ├── Allen_V1_param/                  # BMTK数据目录
        ├── network/                     # 网络结构文件
        ├── components/                  # 模型参数文件
        └── inputs/                      # 输入数据文件
    ├── test_simulation.py        # 主仿真测试脚本
    ├── interactive_test.py       # 交互式测试工具
    ├── bmtk_to_pkl_converter.py  # BMTK格式转换器
    ├── test_visualization.ipynb  # Jupyter可视化notebook
    └── README.md              
```


## 核心模块详解

### 1. test_simulation.py - 主仿真测试模块

#### SparseLayerWithExternalBkg类

**设计动机**: 原始工具包中的`SparseLayer`类使用内部生成的随机噪声来模拟背景输入，这种方式虽然计算效率高，但缺乏生物学真实性。为了支持更真实的背景输入模式，我们开发了`SparseLayerWithExternalBkg`类。

**与原始SparseLayer的关键区别**:

| 特性 | 原始SparseLayer | SparseLayerWithExternalBkg |
|------|----------------|---------------------------|
| **背景输入来源** | 内部生成随机噪声 | 外部真实脉冲数据 |
| **输入参数** | 单一LGN输入 | 分离的LGN和背景输入 |
| **噪声模型** | 泊松随机过程或预计算噪声 | 基于BMTK的真实背景活动 |
| **生物学真实性** | 简化模型 | 高度真实的背景连接 |

**核心功能**:
- **分离处理**: 独立处理LGN输入和背景输入的稀疏矩阵乘法
- **真实连接**: 使用从BMTK数据转换得到的真实背景连接权重
- **动态合并**: 将LGN电流和背景电流动态合并为总输入电流

#### V1SimulationTester类

**功能**: 封装了V1模型的完整仿真测试流程

**核心方法**:

##### `__init__(data_dir, simulation_time, dt, seed)`
- **功能**: 初始化仿真测试器
- **参数**:
  - `data_dir`: 数据目录路径（包含network_dat.pkl和input_dat.pkl）
  - `simulation_time`: 仿真时长（毫秒，默认1000ms）
  - `dt`: 时间步长（毫秒，默认1.0ms）
  - `seed`: 随机种子（确保结果可重复）

##### `load_network_and_input(n_neurons, core_only)`
- **功能**: 加载网络结构和输入数据
- **参数**:
  - `n_neurons`: 使用的神经元数量（None表示使用所有）
  - `core_only`: 是否只使用核心区域神经元（半径<400μm）
- **返回**: 
  - `network`: 网络结构字典，包含神经元参数、连接信息、空间坐标等
  - `input_populations`: 输入信号列表[LGN输入, 背景输入]

**网络结构包含**:
- `n_nodes`: 神经元数量
- `node_params`: 神经元参数（V_th阈值电位、g电导、E_L静息电位等）
- `node_type_ids`: 每个神经元的类型ID
- `synapses`: 突触连接信息（indices、weights、delays）
- `x,y,z`: 神经元的3D空间坐标
- `laminar_indices`: 按层和细胞类型的神经元索引

##### `prepare_simulation(network, input_populations)`
- **功能**: 准备仿真参数，创建BillehColumn神经元模型
- **物理意义**: 配置GLIF3神经元的动力学参数和突触连接
- **返回**: 
  - `cell`: BillehColumn神经元模型
  - `lgn_input`: LGN（外侧膝状体）输入数据
  - `bkg_input`: 背景输入数据

##### `run_simulation(cell, lgn_input, bkg_input, batch_size)`
- **功能**: 执行神经网络仿真
- **算法**: 逐时间步数值积分GLIF3动力学方程
- **核心创新**: 使用`SparseLayerWithExternalBkg`处理真实的背景输入数据
- **仿真流程**:
  1. 准备LGN和背景输入的张量数据
  2. 创建`SparseLayerWithExternalBkg`输入层
  3. 逐时间步计算输入电流和神经元状态
  4. 收集所有时间步的输出数据
- **返回**: 仿真结果字典，包含：
  - `spikes`: 脉冲发放数据 (batch × time × neurons)
  - `voltages`: 膜电位轨迹 (batch × time × neurons)
  - `adaptive_currents`: 自适应电流
  - `psc_rise/psc`: 突触后电流
  - `spike_rates`: 每个神经元的平均发放率

##### `_run_manual_simulation(cell, lgn_spikes, bkg_spikes, lgn_input, bkg_input, batch_size, n_timesteps)`
- **功能**: 核心仿真循环，使用外部背景输入的逐时间步方法
- **技术特点**:
  - **真实背景输入**: 不同于原始工具包的随机噪声，使用真实的背景脉冲数据
  - **分离输入处理**: LGN和背景输入通过不同的稀疏连接矩阵独立处理
  - **动态电流合并**: 每个时间步动态合并LGN电流和背景电流
- **计算优势**: 虽然计算复杂度较高，但提供了更高的生物学真实性
- **调试友好**: 逐时间步的设计便于监控和调试神经元状态变化

##### `save_spikes_to_h5(simulation_results, network, output_file)`
- **功能**: 将仿真结果保存为HDF5格式
- **格式**: 与Allen研究所标准格式兼容
- **结构**: 
  ```
  /spikes/v1/timestamps - 脉冲时间戳 (ms)
  /spikes/v1/node_ids - 神经元节点ID
  ```

### 2. interactive_test.py - 交互式测试模块

#### InteractiveV1Tester类

**功能**: 继承V1SimulationTester，添加交互式功能

**核心方法**:

##### `select_neurons_by_criteria(network, layer, cell_type, spatial_region, neuron_ids)`
- **功能**: 根据多种条件选择神经元
- **选择条件**:
  - `layer`: 皮层层级（'L1', 'L2', 'L3', 'L4', 'L5', 'L6'）
  - `cell_type`: 细胞类型（'e'=兴奋性, 'i'=抑制性）
  - `spatial_region`: 空间区域（x_min, x_max, z_min, z_max）单位微米
  - `neuron_ids`: 直接指定神经元ID列表

##### `analyze_selected_neurons(simulation_results, selected_indices, time_window)`
- **功能**: 分析选定神经元的详细活动
- **分析指标**:
  - **发放率**: 每个神经元的平均发放频率（Hz）
  - **变异系数(CV)**: 衡量发放规律性，CV = σ/μ
  - **同步性指数**: 群体同步程度，反映网络协调性
  - **膜电位统计**: 平均值、标准差、最值等

##### `plot_detailed_activity(simulation_results, selected_indices, analysis)`
- **功能**: 绘制详细的神经活动图
- **图形内容**:
  - **光栅图**: 脉冲发放的时空模式
  - **群体发放率**: 时间演化的群体活动
  - **发放率分布**: 神经元发放率的统计分布
  - **CV分布**: 发放规律性的分布
  - **膜电位轨迹**: 样本神经元的膜电位时间序列

##### `export_neuron_data(simulation_results, neuron_id, output_file)`
- **功能**: 导出单个神经元的详细数据
- **支持格式**: NPZ（NumPy压缩）、CSV
- **数据内容**: 脉冲时间、膜电位、自适应电流等

### 3. bmtk_to_pkl_converter.py - 数据转换模块

#### 功能概述
将BMTK（Brain Modeling Toolkit）格式的网络数据转换为工具包兼容的PKL格式。

#### 核心函数

##### `convert_input_data(bmtk_dir, output_dir)`
- **功能**: 转换输入数据（LGN和背景输入）
- **处理步骤**:
  1. 读取LGN节点信息和脉冲数据
  2. 读取背景节点信息和脉冲数据
  3. 构建连接权重矩阵
  4. 保存为input_dat.pkl格式

**输入数据结构**:
- **LGN输入**: 模拟视觉刺激信号，通常包含方向选择性和时间动态
- **背景输入**: 模拟大脑其他区域的输入，通常为泊松分布的随机脉冲

## 神经科学原理

### GLIF3神经元模型

GLIF3（Generalized Leaky Integrate-and-Fire level 3）是Allen研究所开发的生物学真实神经元模型。

#### 膜电位动力学方程
```
C_m * dV/dt = -g * (V - E_L) + I_syn + I_asc + I_ext
```

**参数物理意义**:
- `C_m`: 膜电容（法拉德），决定膜电位变化的时间常数
- `V`: 膜电位（毫伏）
- `g`: 膜电导（西门子），决定静息状态的膜电阻
- `E_L`: 静息电位（毫伏），神经元的平衡电位
- `I_syn`: 突触电流（安培），来自其他神经元的输入
- `I_asc`: 自适应电流（安培），包含两个分量，模拟钠钾泵等机制
- `I_ext`: 外部输入电流（安培）

#### 自适应电流动力学
```
dI_asc1/dt = -k1 * I_asc1 + A1 * δ(t - t_spike)
dI_asc2/dt = -k2 * I_asc2 + A2 * δ(t - t_spike)
```

**物理意义**: 模拟神经元发放后的自适应过程，包括钠钾泵激活、钙依赖性钾通道开放等。

#### 突触动力学

**双指数突触后电流模型**:
```
I_syn = Σ_i PSC_i(t)
PSC_i(t) = A * (exp(-t/τ_decay) - exp(-t/τ_rise))
```

**四种受体类型**:
1. **AMPA**: 快速兴奋性，τ_rise ≈ 0.2ms, τ_decay ≈ 2ms
2. **NMDA**: 慢速兴奋性，τ_rise ≈ 2ms, τ_decay ≈ 65ms
3. **GABA_A**: 快速抑制性，τ_rise ≈ 0.2ms, τ_decay ≈ 8ms
4. **GABA_B**: 慢速抑制性，τ_rise ≈ 3.5ms, τ_decay ≈ 260ms

### 网络结构

#### 皮层层级组织
- **L1**: 主要包含树突和少量神经元
- **L2/3**: 皮层间连接的主要源头
- **L4**: 接收丘脑输入的主要层级
- **L5**: 皮层输出的主要层级
- **L6**: 反馈到丘脑的主要层级

#### 细胞类型
- **兴奋性神经元**: 释放谷氨酸，激活下游神经元
- **抑制性神经元**: 释放GABA，抑制下游神经元

## 使用指南

### 基本使用

```python
from test_simulation import V1SimulationTester

# 创建测试器
tester = V1SimulationTester(
    data_dir='Allen_V1_param',
    simulation_time=1000,  # 1秒仿真
    dt=1.0,               # 1毫秒时间步长
    seed=42
)

# 加载网络和输入
network, input_populations = tester.load_network_and_input(
    n_neurons=1000,    # 使用1000个神经元
    core_only=True     # 只使用核心区域
)

# 准备仿真
cell, lgn_input, bkg_input = tester.prepare_simulation(network, input_populations)

# 运行仿真
results = tester.run_simulation(cell, lgn_input, bkg_input)
```

### 交互式使用

```python
from interactive_test import InteractiveV1Tester

# 创建交互式测试器
tester = InteractiveV1Tester(data_dir='Allen_V1_param')

# 加载网络
network, input_populations = tester.load_network_and_input()

# 选择特定神经元（例如L4层兴奋性神经元）
selected_indices = tester.select_neurons_by_criteria(
    network, 
    layer='L4', 
    cell_type='e'
)

# 运行仿真
cell, lgn_input, bkg_input = tester.prepare_simulation(network, input_populations)
results = tester.run_simulation(cell, lgn_input, bkg_input)

# 分析选定神经元
analysis = tester.analyze_selected_neurons(results, selected_indices)

# 绘制详细活动图
tester.plot_detailed_activity(results, selected_indices, analysis, 'activity_plot.png')

# 导出特定神经元数据
tester.export_neuron_data(results, neuron_id=100, output_file='neuron_100.npz')
```

### 数据转换

```bash
# 将BMTK格式转换为PKL格式
python bmtk_to_pkl_converter.py Allen_V1_param Converted_param
```

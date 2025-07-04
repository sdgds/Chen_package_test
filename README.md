# V1模型仿真测试工具

## 概述

本测试工具集用于系统性地测试`Training-data-driven-V1-model-test`工具包的仿真功能。该工具包实现了Allen研究所2020年发布的小鼠V1（初级视觉皮层）模型，支持通过反向传播算法修改模型权重。

## 功能特性

### 1. 核心仿真功能
- **GLIF3神经元模型仿真**：实现广义漏积分发放（Generalized Leaky Integrate-and-Fire）神经元动力学
- **突触动力学模拟**：支持4种受体类型的突触动力学
- **网络结构**：基于Allen V1模型的真实解剖学连接
- **输入信号**：支持LGN（外侧膝状体）视觉输入和背景输入

### 2. 分析功能
- **层级分析**：按皮层（L1-L6）和细胞类型（兴奋性/抑制性）分析神经活动
- **空间分析**：神经元空间分布和活动的3D可视化
- **时间动态**：网络活动的时间演化分析
- **连接性分析**：突触连接模式和统计

### 3. 交互式功能
- **神经元选择**：支持按层、类型、空间位置或ID选择特定神经元
- **参数调节**：可自定义仿真时长、时间步长等参数
- **数据导出**：支持导出特定神经元的详细数据（CSV/NPZ格式）

## 技术原理

### GLIF3模型

GLIF3（Generalized Leaky Integrate-and-Fire level 3）模型的膜电位动力学方程：

```
C_m * dV/dt = -g * (V - E_L) + I_syn + I_asc + I_ext
```

其中：
- `C_m`：膜电容
- `V`：膜电位
- `g`：膜电导
- `E_L`：静息电位
- `I_syn`：突触电流
- `I_asc`：自适应电流（包含两个分量）
- `I_ext`：外部输入电流

### 突触动力学

突触后电流（PSC）采用双指数模型：
```
I_syn = Σ_i PSC_i(t)
PSC_i(t) = A * (exp(-t/τ_decay) - exp(-t/τ_rise))
```

支持4种受体类型，每种有不同的时间常数。

### 网络结构

- **神经元数量**：可配置（默认使用核心区域约5000个神经元）
- **层级组织**：6层皮层结构（L1-L6）
- **细胞类型**：兴奋性和抑制性神经元
- **连接性**：基于实验数据的连接概率和权重分布

### 仿真方法详解

#### 方法1：TensorFlow RNN层（原始方法）

```python
# 核心代码
rnn = tf.keras.layers.RNN(cell, return_sequences=True)
outputs = rnn(inputs, initial_state=initial_state)
```

**工作原理：**
1. TensorFlow将整个时间序列作为输入
2. 内部使用优化的C++内核进行计算
3. 自动管理GPU内存和并行计算
4. 支持自动微分进行梯度计算

**优势：**
- 高性能：GPU并行计算，内存优化
- 原生支持：完全兼容TensorFlow生态
- 训练友好：支持反向传播训练

#### 方法2：逐时间步循环（测试方法）

```python
# 核心代码
for t in range(timesteps):
    outputs, state = cell(inputs[:, t, :], state)
    results.append(outputs)
```

**工作原理：**
1. Python循环逐个处理每个时间步
2. 手动调用神经元模型的call()方法
3. 可以在每个时间步插入自定义逻辑
4. 便于监控和调试中间状态

**优势：**
- 透明性：可以观察每个时间步的状态
- 灵活性：易于添加自定义分析和监控
- 调试友好：便于理解和验证算法

**关键等价性证明：**
两种方法调用完全相同的`BillehColumn.call()`方法，实现相同的数学运算：
- GLIF3膜电位微分方程
- 突触后电流动力学
- 自适应电流更新
- 脉冲生成机制




## 使用方法

### 1. 基本仿真测试

```bash
cd test_chen_tools
python test_simulation.py
```

这将运行默认的仿真测试：
- 加载5000个神经元
- 仿真1秒（1000ms）
- 使用逐时间步方法（便于调试）
- 输出各层神经活动统计
- 保存结果到`results/`目录

**两种仿真方法对比：**

| 特性 | 逐时间步方法 | RNN层方法 |
|------|-------------|-----------|
| **计算速度** | 较慢（Python循环） | 快（GPU优化） |
| **内存效率** | 较低 | 高（TensorFlow优化） |
| **调试便利性** | 高（可逐步监控） | 低（黑盒操作） |
| **自定义能力** | 高（可插入自定义逻辑） | 低 |
| **梯度支持** | 复杂 | 完全支持 |
| **适用场景** | 算法调试、详细分析 | 大规模仿真、训练 |

**数学等价性：** 两种方法在数学上是等价的，都实现相同的GLIF3神经元动力学，应产生相同结果（数值精度范围内）。



### 2. 交互式测试

```bash
python interactive_test.py --help
```

常用参数：
- `--neurons`：神经元数量（默认5000）
- `--duration`：仿真时长，单位ms（默认1000）
- `--layer`：选择特定层（L1-L6）
- `--cell-type`：选择细胞类型（e=兴奋性，i=抑制性）
- `--save-plot`：保存图形文件路径
- `--export-neuron`：导出特定神经元ID的数据
- `--export-file`：导出数据文件路径

示例：
```bash
# 分析L4层兴奋性神经元
python interactive_test.py --layer L4 --cell-type e --save-plot L4e_analysis.png

# 导出神经元100的详细数据
python interactive_test.py --export-neuron 100 --export-file neuron_100_data.csv
```

### 3. 方法对比测试

运行两种仿真方法的详细对比：
```bash
python test_rnn_comparison.py
```

该测试将：
- 使用相同数据同时运行两种方法
- 对比计算速度和准确性
- 生成详细的对比图表
- 验证数学等价性

### 4. Jupyter Notebook可视化

启动Jupyter并打开`test_visualization.ipynb`：
```bash
jupyter notebook test_visualization.ipynb
```

Notebook包含：
- 网络加载和仿真运行
- 层级活动可视化
- 空间组织3D可视化
- 时间动态分析
- 网络连接性分析

## 输出文件

### 仿真结果文件
- `simulation_results_YYYYMMDD_HHMMSS.npz`：包含完整的仿真数据
  - `spikes`：脉冲发放数据 (batch × time × neurons)
  - `voltages`：膜电位数据 (batch × time × neurons)
  - `adaptive_currents`：自适应电流
  - `psc_rise/psc`：突触后电流
  - `spike_rates`：每个神经元的平均发放率

### 分析结果文件
- `analysis_results_YYYYMMDD_HHMMSS.pkl`：包含统计分析结果
  - 各层各类型神经元的统计信息
  - 平均发放率、膜电位统计等

### 图形文件
- `sample_neuron_activity.png`：样本神经元活动图
- 自定义保存的分析图形

## 代码结构

```
test_chen_tools/
├── test_simulation.py        # 主测试脚本（支持两种方法）
├── interactive_test.py       # 交互式测试脚本
├── test_rnn_comparison.py    # RNN层vs逐时间步对比测试
├── test_visualization.ipynb  # Jupyter可视化notebook
├── run_all_tests.py         # 自动化测试脚本
├── __init__.py              # Python包初始化文件
├── README.md                # 本文档
├── results/                 # 仿真结果目录
└── figures/                 # 图形输出目录
```

## 关键类和函数

### V1SimulationTester类
主要的测试类，提供以下方法：
- `load_network_and_input()`: 加载网络结构和输入数据
- `prepare_simulation()`: 准备仿真参数
- `run_simulation()`: 运行神经网络仿真
- `analyze_by_layer_and_type()`: 按层和类型分析结果
- `save_results()`: 保存仿真结果
- `plot_sample_activity()`: 绘制样本活动

### InteractiveV1Tester类
扩展了V1SimulationTester，添加：
- `select_neurons_by_criteria()`: 根据条件选择神经元
- `analyze_selected_neurons()`: 分析选定神经元
- `plot_detailed_activity()`: 绘制详细活动图
- `export_neuron_data()`: 导出神经元数据

## 注意事项

1. **内存需求**：仿真大规模网络需要充足的内存（建议16GB以上）
2. **计算时间**：仿真时间与神经元数量和仿真时长成正比
3. **GPU支持**：如果安装了GPU版本的TensorFlow，将自动使用GPU加速
4. **数据文件**：需要GLIF_network文件夹中的网络数据文件

## 扩展和自定义

### 添加新的分析功能
可以继承`V1SimulationTester`类并添加新方法：

```python
class MyCustomTester(V1SimulationTester):
    def my_analysis(self, simulation_results):
        # 自定义分析代码
        pass
```

### 修改仿真参数
在创建BillehColumn时可以调整参数：
- `gauss_std`: 脉冲梯度的标准差
- `dampening_factor`: 脉冲梯度的阻尼因子
- `max_delay`: 最大突触延迟

## 参考文献

1. Billeh et al. (2020). "Systematic integration of structural and functional data into multi-scale models of mouse primary visual cortex." Neuron.
2. Allen Institute for Brain Science. "Visual Coding - Neuropixels" dataset.

## 联系和支持

如有问题或建议，请参考主工具包的文档或联系相关维护者。 
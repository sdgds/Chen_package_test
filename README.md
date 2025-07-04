# V1神经网络模型仿真测试工具包

本工具包提供了一套完整的解决方案，用于测试和验证基于Allen研究所小鼠V1（初级视觉皮层）模型的神经网络仿真功能。支持BMTK格式数据转换、多种仿真方法测试、交互式分析和可视化。

## 🚀 主要功能

- **数据转换**: 将BMTK格式数据转换为PKL格式，便于Chen工具包导入和仿真
- **神经网络仿真**: 基于GLIF3神经元模型的网络仿真
- **交互式分析**: 按层级、细胞类型或空间位置选择和分析特定神经元

## 📋 目录

- [工具概述](#-工具概述)
- [快速开始](#-快速开始)
- [工具详解](#-工具详解)

## 🔧 工具概述

### 1. 数据转换工具 (`bmtk_to_pkl_converter.py`)

**功能**: 将BMTK（Brain Modeling Toolkit）格式的V1模型数据转换为仿真所需的PKL格式。

**输入**: BMTK格式数据（lgn和bkg输入spike，网络连接参数）  
**输出**: `input_dat.pkl` 文件
**注意**: `network_dat.pkl`使用Chen工具包中自带的

### 2. 仿真测试工具 (`test_simulation.py`)

**功能**: 主要的仿真测试脚本，支持大规模神经网络仿真。

**特色**: 
- 两种仿真方法：TensorFlow RNN层（高效）和逐时间步循环（易调试）
- 支持自定义仿真参数（时长、时间步长、神经元数量等）
- 按层级和细胞类型进行结果分析

### 3. 交互式测试工具 (`interactive_test.py`)

**功能**: 在仿真测试基础上增加交互功能，支持精细化分析。

**特色**:
- 按条件选择特定神经元群体
- 详细的活动分析（发放率、同步性、膜电位统计等）
- 数据导出功能（CSV/NPZ格式）

### 4. 可视化工具 (`test_visualization.ipynb`)

**功能**: Jupyter Notebook格式的可视化工具。

**特色**:
- 网络结构3D可视化
- 神经活动动态展示
- 交互式图表和分析

### 数据准备

确保您有以下格式的数据文件：

```
Training-data-driven-V1-model-test
├── Chen工具包的所有内容
    Chen_package_test (我们的测试工具包放在Chen工具包文件夹下)
    ├── Allen_V1_param/                  # BMTK数据目录
        ├── network/                     # 网络结构文件
        ├── components/                  # 模型参数文件
        └── inputs/                      # 输入数据文件
    ├── 测试工具包的所有文件
```

## 🚀 快速开始

### 1. 数据转换（如果您有BMTK格式数据）

```bash
# 进入Chen_package_test文件夹，转换BMTK数据为PKL格式。这会自动把BMTK的lgn和bkg信号转换为pkl格式。
python bmtk_to_pkl_converter.py Allen_V1_param Allen_V1_param
```

### 2. 基础仿真测试

```bash
# 使用默认参数运行仿真
python test_simulation.py Allen_V1_param results

# 查看结果
ls results/
# simulation_results_20240101_120000.npz
# analysis_results_20240101_120000.pkl
```

### 3. 交互式分析和可视化

```bash
# 启动Jupyter Notebook
jupyter notebook test_visualization.ipynb
```

## 📖 工具详解

### bmtk_to_pkl_converter.py

```bash
# 基本语法
python bmtk_to_pkl_converter.py <input_dir> [output_dir]

# 示例
python bmtk_to_pkl_converter.py Allen_V1_param ./converted_data
```

**主要功能**:
- 转换网络结构（神经元参数、连接权重）
- 转换输入数据（LGN视觉输入、背景输入）

### test_simulation.py

```bash
# 基本语法
python test_simulation.py <data_dir> <output_dir> [选项]

# 常用选项
--simulation-time 1000     # 仿真时长（毫秒）
--dt 1.0                   # 时间步长（毫秒）
--n-neurons 5000          # 神经元数量
--core-only                # 仅使用核心区域神经元
--use-rnn-layer           # 使用RNN层方法（更快）
--plot-activity           # 生成活动图
```

### interactive_test.py

**分析功能**:
- 发放率统计
- 变异系数（CV）分析
- 同步性指数
- 膜电位统计
# EEG数据63维到64维转换工具

## 简介

本工具实现了与MATLAB代码完全一致的EEG数据通道转换功能，将63通道数据转换为64通道（恢复缺失的Cz通道）。**严格保证与MATLAB逻辑一致，确保实验可复现**。

## 文件说明

- `eeg_channel_conversion.py`: 核心转换模块
- `test_conversion.py`: 测试脚本
- `EEG_Channel_Conversion_Documentation.md`: 详细技术文档
- `README_CN.md`: 本文件（使用说明）

## 快速开始

### 1. 安装依赖

```bash
pip install numpy scipy
```

### 2. 基本使用

```python
from eeg_channel_conversion import num_ch_corr
import numpy as np

# 加载63通道数据（示例）
# signal_63ch 应该是形状为 (63, time_points) 的numpy数组
signal_63ch = np.random.randn(63, 1000).astype(np.float64)

# 转换为64通道
signal_64ch = num_ch_corr(signal_63ch)

print(f"输入维度: {signal_63ch.shape}")  # (63, 1000)
print(f"输出维度: {signal_64ch.shape}")  # (64, 1000)
```

### 3. 加载和处理真实数据

```python
from eeg_channel_conversion import load_creativity_data, convert_all_channels

# 加载参与者1的数据
data_63ch = load_creativity_data(".", subject_id=1)

# 转换为64通道
data_64ch = convert_all_channels(data_63ch)

# 查看结果
for state, signal in data_64ch.items():
    print(f"{state}: {signal.shape}")
```

### 4. 批量处理所有参与者

```python
from eeg_channel_conversion import process_all_subjects

# 处理所有参与者（1-28）
all_data = process_all_subjects(
    data_dir=".",
    save_output=True,
    output_dir="./output_64ch"
)
```

## 转换逻辑

### 核心步骤

1. **计算Cz通道**：使用8个相邻通道的平均值
   ```
   Cz = (FC1 + FCz + FC2 + C1 + C2 + CP1 + CPz + CP2) / 8
   ```

2. **重组通道**：
   - 前23个通道保持不变
   - 第24位插入插值得到的Cz
   - 原24-63通道映射到25-64位置

### 与MATLAB的一致性

本实现与MATLAB代码 `Num_Ch_Corr.m` 完全一致：

- ✅ 相同的Cz插值公式
- ✅ 相同的通道映射逻辑
- ✅ 相同的数值精度（float64）
- ✅ 相同的输出维度

## 验证

运行测试脚本验证转换正确性：

```bash
python test_conversion.py
```

测试包括：
- 基本转换功能
- 数值精度验证
- 边界情况测试
- 真实数据测试（如果可用）
- 与MATLAB结果对比（如果可用）

## 数据格式

### 输入数据

- **文件路径**: `./Creativity_EEG_Dataset/Data_Creativity_Sub_{subject_id}.mat`
- **变量命名**: 
  - 任务状态: `Creativity_{subject_id}_{trial}_{state}`
  - 休息状态: `Creativity_{subject_id}_RST1` 或 `RST2`
- **数据维度**: `63 × 时间点数`

### 输出数据

- **数据维度**: `64 × 时间点数`
- **通道顺序**: 符合BrainVision标准64通道映射

## 函数说明

### `num_ch_corr(signal_63ch)`

核心转换函数，将63通道转换为64通道。

**参数**:
- `signal_63ch`: numpy数组，形状为 `(63, time_points)`

**返回**:
- numpy数组，形状为 `(64, time_points)`

### `load_creativity_data(data_dir, subject_id)`

加载指定参与者的EEG数据。

**参数**:
- `data_dir`: 数据目录路径
- `subject_id`: 参与者编号（1-28）

**返回**:
- 字典，包含所有状态的63通道数据

### `convert_all_channels(data_dict)`

批量转换字典中的所有数据。

**参数**:
- `data_dict`: 包含63通道数据的字典

**返回**:
- 包含64通道数据的字典

### `verify_conversion(original_63ch, converted_64ch, tolerance=1e-10)`

验证转换结果的正确性。

**参数**:
- `original_63ch`: 原始63通道数据
- `converted_64ch`: 转换后的64通道数据
- `tolerance`: 数值容差

**返回**:
- `(is_valid, report)`: 验证结果和详细报告

## 注意事项

1. **索引差异**: MATLAB使用1-based索引，Python使用0-based索引，代码中已正确处理
2. **数据类型**: 使用`float64`（double）精度，与MATLAB保持一致
3. **数值精度**: 浮点运算可能存在微小差异（< 1e-10），这在可接受范围内
4. **内存管理**: 对于大型数据集，注意内存使用

## 常见问题

### Q: 转换后的数据与MATLAB结果有微小差异？

A: 这是正常的。浮点运算在不同平台可能有微小差异（通常 < 1e-10）。如果差异较大，请检查：
- 数据类型是否为float64
- 输入数据是否完全相同
- 是否有额外的数据处理步骤

### Q: 如何验证转换正确性？

A: 使用 `verify_conversion()` 函数，它会检查：
- 维度是否正确
- 前23个通道是否一致
- Cz通道计算是否正确
- 后40个通道映射是否正确

### Q: 可以处理其他格式的数据吗？

A: 只要数据是 `(63, time_points)` 格式的numpy数组，就可以使用 `num_ch_corr()` 函数转换。

## 技术支持

如有问题，请参考：
- `EEG_Channel_Conversion_Documentation.md`: 详细技术文档
- `test_conversion.py`: 测试示例

## 版本历史

- v1.0: 初始版本，与MATLAB代码完全一致


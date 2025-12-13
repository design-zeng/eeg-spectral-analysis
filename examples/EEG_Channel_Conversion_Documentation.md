# EEG数据63维到64维转换文档

## 一、背景说明

### 1.1 数据来源
- **原始数据格式**：63通道 × 时间点
- **原因**：Cz通道（第24个通道）在EEG记录和预处理阶段被移除
- **目标**：恢复标准64通道格式，用于后续分析和可视化

### 1.2 数据文件结构
- **文件路径**：`./Creativity_EEG_Dataset/Data_Creativity_Sub_{subject_id}.mat`
- **变量命名规则**：
  - 任务状态：`Creativity_{subject_id}_{trial}_{state}`
    - `state` 可以是：`IDG`（Idea Generation）、`IDE`（Idea Evolution）、`IDR`（Idea Rating）
    - `trial` 可以是：1, 2, 3
  - 休息状态：`Creativity_{subject_id}_RST1` 或 `Creativity_{subject_id}_RST2`
- **数据维度**：所有变量均为 `63 × 时间点数` 的double数组

## 二、转换逻辑详解

### 2.1 标准64通道映射（BrainVision标准）

根据BrainVision EEG记录系统的标准64通道映射：

| 通道位置 | 通道名称 | 说明 |
|---------|---------|------|
| 1-23 | Fp1, Fz, F3, F7, FT9, FC5, FC1, C3, T7, TP9, CP5, CP1, Pz, P3, P7, O1, Oz, O2, P4, P8, TP10, CP6, CP2 | 保持不变 |
| **24** | **Cz** | **需要插值恢复** |
| 25-64 | C4, T8, FT10, FC6, FC2, F4, F8, Fp2, AF7, AF3, AFz, F1, F5, FT7, FC3, FCz, C1, C5, TP7, CP3, P1, P5, PO7, PO3, POz, PO4, PO8, P6, P2, CPz, CP4, TP8, C6, C2, FC4, FT8, F6, F2, AF4, AF8 | 原24-63通道向后移动 |

### 2.2 Cz通道插值方法

**插值公式**（与MATLAB完全一致）：
```
CZ = (Ch7 + Ch39 + Ch28 + Ch40 + Ch57 + Ch12 + Ch53 + Ch23) / 8
```

**使用的8个相邻通道**（在63通道原始数据中的位置）：
- Ch7: FC1
- Ch39: FCz  
- Ch28: FC2
- Ch40: C1
- Ch57: C2
- Ch12: CP1
- Ch53: CPz
- Ch23: CP2

**注意**：这些通道索引是基于**63通道原始数据**的位置，不是64通道标准位置。

### 2.3 转换步骤（与MATLAB `Num_Ch_Corr.m`完全一致）

#### 步骤1：计算Cz通道
```python
# 从63通道数据中提取8个相邻通道
CZ = (Signal2[6] + Signal2[38] + Signal2[27] + Signal2[39] + 
      Signal2[56] + Signal2[11] + Signal2[52] + Signal2[22]) / 8.0
```
**注意**：Python使用0-based索引，MATLAB使用1-based索引

#### 步骤2：重组通道
```python
# 1. 保留前23个通道（索引0-22）
output[0:23, :] = input[0:23, :]

# 2. 在第24位（索引23）插入插值得到的Cz
output[23, :] = CZ

# 3. 将原24-63通道（索引23-62）映射到25-64（索引24-63）
output[24:64, :] = input[23:63, :]
```

### 2.4 索引映射表

| MATLAB索引（1-based） | Python索引（0-based） | 原始63通道位置 | 新64通道位置 | 通道名称 |
|---------------------|---------------------|--------------|------------|---------|
| 1-23 | 0-22 | 1-23 | 1-23 | Fp1...CP2 |
| 24 | 23 | - | 24 | **Cz（插值）** |
| 25-64 | 24-63 | 24-63 | 25-64 | C4...AF8 |

## 三、MATLAB原始代码参考

### 3.1 核心转换函数（`Num_Ch_Corr.m`）

```matlab
function [Sig_IDG_out] = Num_Ch_Corr(Sig_IDG)
    Signal2 = Sig_IDG;  % 输入：63通道 × 时间点
    
    % 计算Cz通道（8个相邻通道的平均值）
    CZ = (Signal2(7,:) + Signal2(39,:) + Signal2(28,:) + Signal2(40,:) + ...
          Signal2(57,:) + Signal2(12,:) + Signal2(53,:) + Signal2(23,:)) ./ 8;
    
    % 重组通道
    Sig1(1:23,:) = Signal2(1:23,:);      % 前23个通道保持不变
    Sig1(24,:) = CZ;                       % 第24位插入Cz
    Sig1(25:64,:) = Signal2(24:63,:);     % 原24-63通道映射到25-64
    
    Sig_IDG_out = Sig1;  % 输出：64通道 × 时间点
end
```

### 3.2 数据加载（`Read_Signals.m`）

```matlab
for k = 1:28  % 28个参与者
    filename = sprintf('./Creativity_EEG_Dataset/Data_Creativity_Sub_%d.mat', k);
    load(filename);
    
    for ii = 1:3  % 3个试验
        % 提取不同状态的信号
        Signal_IDG = eval(strcat('Creativity_', num2str(k), '_', num2str(ii), '_IDG'));
        Signal_IDE = eval(strcat('Creativity_', num2str(k), '_', num2str(ii), '_IDE'));
        Signal_IDR = eval(strcat('Creativity_', num2str(k), '_', num2str(ii), '_IDR'));
        
        % 应用通道转换
        [Sig_IDG_out] = Num_Ch_Corr(Signal_IDG);
        [Sig_IDE_out] = Num_Ch_Corr(Signal_IDE);
        [Sig_IDR_out] = Num_Ch_Corr(Signal_IDR);
    end
    
    % 处理休息状态
    Signal_RST1 = eval(strcat('Creativity_', num2str(k), '_RST1'));
    Signal_RST2 = eval(strcat('Creativity_', num2str(k), '_RST2'));
    
    [Sig_RST1_out] = Num_Ch_Corr(Signal_RST1);
    [Sig_RST2_out] = Num_Ch_Corr(Signal_RST2);
end
```

## 四、验证方法

### 4.1 维度验证
- **输入**：63 × 时间点数
- **输出**：64 × 时间点数
- **时间点数**：必须保持不变

### 4.2 数值验证
1. **前23个通道**：应与原始数据完全一致
2. **Cz通道**：应为8个指定通道的平均值
3. **后40个通道**：应与原始数据的24-63通道完全一致

### 4.3 与MATLAB结果对比
建议使用相同的数据文件，分别用MATLAB和Python处理，然后对比结果：
- 使用 `numpy.allclose()` 进行数值比较（允许小的浮点误差）
- 检查所有通道的数值差异是否在可接受范围内（通常 < 1e-10）

## 五、注意事项

1. **索引差异**：MATLAB使用1-based索引，Python使用0-based索引，转换时需特别注意
2. **数据类型**：确保使用`float64`（double）精度，与MATLAB保持一致
3. **内存管理**：对于大型数据集，注意内存使用
4. **数值精度**：浮点运算可能存在微小差异，但应在可接受范围内（< 1e-10）

## 六、通道名称对照表

### 6.1 63通道原始顺序（输入）
1. Fp1, 2. Fz, 3. F3, 4. F7, 5. FT9, 6. FC5, 7. FC1, 8. C3, 9. T7, 
10. TP9, 11. CP5, 12. CP1, 13. Pz, 14. P3, 15. P7, 16. O1, 17. Oz, 
18. O2, 19. P4, 20. P8, 21. TP10, 22. CP6, 23. CP2, 
**（Cz缺失）**, 
24. C4, 25. T8, 26. FT10, 27. FC6, 28. FC2, 29. F4, 30. F8, 31. Fp2, 
32. AF7, 33. AF3, 34. AFz, 35. F1, 36. F5, 37. FT7, 38. FC3, 39. FCz, 
40. C1, 41. C5, 42. TP7, 43. CP3, 44. P1, 45. P5, 46. PO7, 47. PO3, 
48. POz, 49. PO4, 50. PO8, 51. P6, 52. P2, 53. CPz, 54. CP4, 55. TP8, 
56. C6, 57. C2, 58. FC4, 59. FT8, 60. F6, 61. F2, 62. AF4, 63. AF8

### 6.2 64通道标准顺序（输出）
1-23同上，**24. Cz（插值）**，25-64对应原24-63通道


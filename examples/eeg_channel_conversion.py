"""
EEG数据63维到64维转换模块
与MATLAB代码逻辑完全一致，确保实验可复现

作者：基于MATLAB代码 Num_Ch_Corr.m 和 Read_Signals.m 转换
日期：2024
"""

import numpy as np
import scipy.io as sio
import os
from typing import Dict, Tuple, Optional
import warnings


def num_ch_corr(signal_63ch: np.ndarray) -> np.ndarray:
    """
    将63通道EEG数据转换为64通道（插入Cz通道）
    
    此函数与MATLAB的 Num_Ch_Corr.m 函数逻辑完全一致
    
    参数:
        signal_63ch: numpy数组，形状为 (63, time_points)
                    63通道的EEG数据，Cz通道缺失
    
    返回:
        signal_64ch: numpy数组，形状为 (64, time_points)
                    64通道的EEG数据，Cz通道已通过插值恢复
    
    转换逻辑:
        1. 使用8个相邻通道的平均值计算Cz通道
        2. 前23个通道保持不变
        3. 在第24位插入插值得到的Cz
        4. 原24-63通道映射到25-64位置
    
    注意:
        - 输入必须是63通道
        - 输出为64通道
        - 时间维度保持不变
    """
    # 验证输入维度
    if signal_63ch.ndim != 2:
        raise ValueError(f"输入信号必须是2维数组 (channels, time_points)，当前维度: {signal_63ch.ndim}")
    
    num_channels, num_timepoints = signal_63ch.shape
    
    if num_channels != 63:
        raise ValueError(f"输入必须是63通道，当前通道数: {num_channels}")
    
    # 确保数据类型为float64（与MATLAB double一致）
    signal_63ch = signal_63ch.astype(np.float64)
    
    # ============================================================
    # 步骤1：计算Cz通道（与MATLAB完全一致）
    # ============================================================
    # MATLAB索引（1-based）-> Python索引（0-based）映射：
    # Ch7  -> index 6  (FC1)
    # Ch39 -> index 38 (FCz)
    # Ch28 -> index 27 (FC2)
    # Ch40 -> index 39 (C1)
    # Ch57 -> index 56 (C2)
    # Ch12 -> index 11 (CP1)
    # Ch53 -> index 52 (CPz)
    # Ch23 -> index 22 (CP2)
    
    # 提取8个相邻通道
    ch7  = signal_63ch[6, :]   # FC1
    ch39 = signal_63ch[38, :]  # FCz
    ch28 = signal_63ch[27, :]  # FC2
    ch40 = signal_63ch[39, :]  # C1
    ch57 = signal_63ch[56, :]  # C2
    ch12 = signal_63ch[11, :]  # CP1
    ch53 = signal_63ch[52, :]  # CPz
    ch23 = signal_63ch[22, :]  # CP2
    
    # 计算Cz通道（8个通道的平均值）
    # 与MATLAB代码完全一致：CZ = (Signal2(7,:)+Signal2(39,:)+...)/8
    cz_channel = (ch7 + ch39 + ch28 + ch40 + ch57 + ch12 + ch53 + ch23) / 8.0
    
    # ============================================================
    # 步骤2：重组通道（与MATLAB完全一致）
    # ============================================================
    # 初始化64通道输出数组
    signal_64ch = np.zeros((64, num_timepoints), dtype=np.float64)
    
    # 1. 保留前23个通道（MATLAB: Sig1(1:23,:) = Signal2(1:23,:)）
    # Python索引0-22对应MATLAB索引1-23
    signal_64ch[0:23, :] = signal_63ch[0:23, :]
    
    # 2. 在第24位插入Cz（MATLAB: Sig1(24,:) = CZ）
    # Python索引23对应MATLAB索引24
    signal_64ch[23, :] = cz_channel
    
    # 3. 将原24-63通道映射到25-64（MATLAB: Sig1(25:64,:) = Signal2(24:63,:)）
    # Python索引23-62对应MATLAB索引24-63，映射到输出索引24-63
    signal_64ch[24:64, :] = signal_63ch[23:63, :]
    
    return signal_64ch


def load_creativity_data(data_dir: str, subject_id: int) -> Dict[str, np.ndarray]:
    """
    加载指定参与者的EEG数据
    
    参数:
        data_dir: 数据目录路径（包含Creativity_EEG_Dataset文件夹）
        subject_id: 参与者编号（1-28）
    
    返回:
        data_dict: 字典，包含所有状态的EEG数据
                  - 键：状态名称（如'IDG_1', 'IDE_2', 'RST1'等）
                  - 值：numpy数组，形状为(63, time_points)
    """
    # 构建文件路径
    filename = os.path.join(data_dir, 'Creativity_EEG_Dataset', 
                           f'Data_Creativity_Sub_{subject_id}.mat')
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"数据文件不存在: {filename}")
    
    # 加载MAT文件
    mat_data = sio.loadmat(filename)
    
    # 提取所有相关变量
    data_dict = {}
    
    # 提取任务状态数据（3个试验 × 3种状态）
    for trial in [1, 2, 3]:
        for state in ['IDG', 'IDE', 'IDR']:
            var_name = f'Creativity_{subject_id}_{trial}_{state}'
            if var_name in mat_data:
                # 移除MATLAB的元数据键
                if not var_name.startswith('__'):
                    data_dict[f'{state}_{trial}'] = mat_data[var_name]
    
    # 提取休息状态数据
    for rst in ['RST1', 'RST2']:
        var_name = f'Creativity_{subject_id}_{rst}'
        if var_name in mat_data:
            if not var_name.startswith('__'):
                data_dict[rst] = mat_data[var_name]
    
    return data_dict


def convert_all_channels(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    将字典中所有63通道数据转换为64通道
    
    参数:
        data_dict: 包含63通道数据的字典
    
    返回:
        converted_dict: 包含64通道数据的字典
    """
    converted_dict = {}
    
    for key, signal_63ch in data_dict.items():
        # 验证数据维度
        if signal_63ch.ndim == 2 and signal_63ch.shape[0] == 63:
            converted_dict[key] = num_ch_corr(signal_63ch)
        else:
            warnings.warn(f"跳过 {key}：维度不符合要求 {signal_63ch.shape}")
    
    return converted_dict


def process_all_subjects(data_dir: str, subject_ids: Optional[list] = None, 
                        save_output: bool = False, output_dir: Optional[str] = None) -> Dict:
    """
    处理所有参与者的数据
    
    参数:
        data_dir: 数据目录路径
        subject_ids: 要处理的参与者编号列表，None表示处理所有（1-28）
        save_output: 是否保存转换后的数据
        output_dir: 输出目录（如果save_output=True）
    
    返回:
        all_data: 嵌套字典，结构为 {subject_id: {state: 64ch_data}}
    """
    if subject_ids is None:
        subject_ids = list(range(1, 29))  # 1-28
    
    all_data = {}
    
    for subject_id in subject_ids:
        print(f"处理参与者 {subject_id}...")
        
        try:
            # 加载数据
            data_63ch = load_creativity_data(data_dir, subject_id)
            
            # 转换为64通道
            data_64ch = convert_all_channels(data_63ch)
            
            all_data[subject_id] = data_64ch
            
            # 保存数据（如果需要）
            if save_output and output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, 
                                         f'Data_Creativity_Sub_{subject_id}_64ch.mat')
                
                # 准备保存的数据（使用原始变量名）
                save_dict = {}
                for key, value in data_64ch.items():
                    # 重构变量名
                    if key in ['RST1', 'RST2']:
                        var_name = f'Creativity_{subject_id}_{key}'
                    else:
                        state, trial = key.split('_')
                        var_name = f'Creativity_{subject_id}_{trial}_{state}'
                    save_dict[var_name] = value
                
                sio.savemat(output_file, save_dict)
                print(f"  已保存到: {output_file}")
            
        except Exception as e:
            print(f"  处理参与者 {subject_id} 时出错: {e}")
            continue
    
    return all_data


def verify_conversion(original_63ch: np.ndarray, converted_64ch: np.ndarray, 
                     tolerance: float = 1e-10) -> Tuple[bool, Dict]:
    """
    验证转换结果的正确性
    
    参数:
        original_63ch: 原始63通道数据
        converted_64ch: 转换后的64通道数据
        tolerance: 数值容差
    
    返回:
        is_valid: 是否通过验证
        report: 验证报告字典
    """
    report = {}
    
    # 1. 维度检查
    expected_shape = (64, original_63ch.shape[1])
    shape_valid = converted_64ch.shape == expected_shape
    report['维度检查'] = {
        '通过': shape_valid,
        '期望': expected_shape,
        '实际': converted_64ch.shape
    }
    
    # 2. 前23个通道检查
    channels_1_23_valid = np.allclose(
        converted_64ch[0:23, :], 
        original_63ch[0:23, :], 
        atol=tolerance
    )
    max_diff_1_23 = np.max(np.abs(converted_64ch[0:23, :] - original_63ch[0:23, :]))
    report['前23通道检查'] = {
        '通过': channels_1_23_valid,
        '最大差异': max_diff_1_23
    }
    
    # 3. Cz通道检查
    # 重新计算Cz
    ch7  = original_63ch[6, :]
    ch39 = original_63ch[38, :]
    ch28 = original_63ch[27, :]
    ch40 = original_63ch[39, :]
    ch57 = original_63ch[56, :]
    ch12 = original_63ch[11, :]
    ch53 = original_63ch[52, :]
    ch23 = original_63ch[22, :]
    expected_cz = (ch7 + ch39 + ch28 + ch40 + ch57 + ch12 + ch53 + ch23) / 8.0
    
    cz_valid = np.allclose(converted_64ch[23, :], expected_cz, atol=tolerance)
    max_diff_cz = np.max(np.abs(converted_64ch[23, :] - expected_cz))
    report['Cz通道检查'] = {
        '通过': cz_valid,
        '最大差异': max_diff_cz
    }
    
    # 4. 后40个通道检查（原24-63映射到25-64）
    channels_25_64_valid = np.allclose(
        converted_64ch[24:64, :], 
        original_63ch[23:63, :], 
        atol=tolerance
    )
    max_diff_25_64 = np.max(np.abs(converted_64ch[24:64, :] - original_63ch[23:63, :]))
    report['后40通道检查'] = {
        '通过': channels_25_64_valid,
        '最大差异': max_diff_25_64
    }
    
    # 综合验证结果
    is_valid = (shape_valid and channels_1_23_valid and 
                cz_valid and channels_25_64_valid)
    
    report['总体验证'] = {
        '通过': is_valid,
        '容差': tolerance
    }
    
    return is_valid, report


# ============================================================
# 示例使用代码
# ============================================================

if __name__ == "__main__":
    # 示例1：单个信号转换
    print("=" * 60)
    print("示例1：单个信号转换")
    print("=" * 60)
    
    # 创建模拟的63通道数据（用于测试）
    np.random.seed(42)
    test_signal_63ch = np.random.randn(63, 1000).astype(np.float64)
    
    # 转换为64通道
    test_signal_64ch = num_ch_corr(test_signal_63ch)
    
    print(f"输入维度: {test_signal_63ch.shape}")
    print(f"输出维度: {test_signal_64ch.shape}")
    
    # 验证转换
    is_valid, report = verify_conversion(test_signal_63ch, test_signal_64ch)
    print(f"\n验证结果: {'通过' if is_valid else '失败'}")
    for check_name, check_result in report.items():
        print(f"  {check_name}: {check_result}")
    
    # 示例2：加载和处理真实数据
    print("\n" + "=" * 60)
    print("示例2：加载和处理真实数据")
    print("=" * 60)
    
    # 注意：需要根据实际路径修改
    data_directory = "."  # 修改为实际的数据目录
    
    try:
        # 加载参与者1的数据
        data_63ch = load_creativity_data(data_directory, subject_id=1)
        print(f"\n加载的数据状态: {list(data_63ch.keys())}")
        
        # 显示每个状态的维度
        for state, signal in data_63ch.items():
            print(f"  {state}: {signal.shape}")
        
        # 转换为64通道
        data_64ch = convert_all_channels(data_63ch)
        print(f"\n转换后的数据状态: {list(data_64ch.keys())}")
        
        # 显示转换后的维度
        for state, signal in data_64ch.items():
            print(f"  {state}: {signal.shape}")
        
        # 验证第一个状态的转换
        first_state = list(data_63ch.keys())[0]
        is_valid, report = verify_conversion(
            data_63ch[first_state], 
            data_64ch[first_state]
        )
        print(f"\n{first_state} 验证结果: {'通过' if is_valid else '失败'}")
        
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")
        print("请确保数据文件路径正确")
    
    # 示例3：批量处理所有参与者
    print("\n" + "=" * 60)
    print("示例3：批量处理（注释掉，需要时取消注释）")
    print("=" * 60)
    print("# 取消下面的注释以批量处理所有参与者：")
    print("# all_data = process_all_subjects(data_directory, save_output=True, output_dir='./output_64ch')")


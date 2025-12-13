"""
EEG通道转换使用示例

本文件展示了如何使用eeg_channel_conversion模块进行数据转换
"""

import numpy as np
from eeg_channel_conversion import (
    num_ch_corr, 
    load_creativity_data, 
    convert_all_channels,
    process_all_subjects,
    verify_conversion
)


def example_1_single_signal():
    """示例1：转换单个信号"""
    print("=" * 60)
    print("示例1：转换单个63通道信号为64通道")
    print("=" * 60)
    
    # 创建模拟的63通道数据
    # 在实际使用中，这应该是从文件加载的真实数据
    np.random.seed(42)
    signal_63ch = np.random.randn(63, 1000).astype(np.float64)
    
    print(f"输入数据维度: {signal_63ch.shape}")
    
    # 执行转换
    signal_64ch = num_ch_corr(signal_63ch)
    
    print(f"输出数据维度: {signal_64ch.shape}")
    print("✓ 转换完成\n")


def example_2_load_and_convert():
    """示例2：加载真实数据并转换"""
    print("=" * 60)
    print("示例2：加载真实数据并转换")
    print("=" * 60)
    
    # 数据目录（根据实际情况修改）
    data_dir = "."
    subject_id = 1
    
    try:
        # 加载数据
        print(f"正在加载参与者 {subject_id} 的数据...")
        data_63ch = load_creativity_data(data_dir, subject_id)
        
        print(f"✓ 成功加载，包含以下状态: {list(data_63ch.keys())}")
        
        # 显示每个状态的维度
        print("\n原始数据维度（63通道）:")
        for state, signal in data_63ch.items():
            print(f"  {state:10s}: {signal.shape}")
        
        # 转换为64通道
        print("\n正在转换为64通道...")
        data_64ch = convert_all_channels(data_63ch)
        
        print("✓ 转换完成")
        print("\n转换后数据维度（64通道）:")
        for state, signal in data_64ch.items():
            print(f"  {state:10s}: {signal.shape}")
        
        # 验证转换结果
        print("\n验证转换结果...")
        first_state = list(data_63ch.keys())[0]
        is_valid, report = verify_conversion(
            data_63ch[first_state], 
            data_64ch[first_state]
        )
        
        if is_valid:
            print(f"✓ {first_state} 验证通过")
        else:
            print(f"✗ {first_state} 验证失败")
            print("  详细报告:", report)
        
    except FileNotFoundError as e:
        print(f"✗ 数据文件未找到: {e}")
        print("  请确保数据文件路径正确")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
    
    print()


def example_3_batch_processing():
    """示例3：批量处理多个参与者"""
    print("=" * 60)
    print("示例3：批量处理多个参与者")
    print("=" * 60)
    
    data_dir = "."
    
    # 处理前3个参与者（示例）
    subject_ids = [1, 2, 3]
    
    print(f"正在处理参与者: {subject_ids}")
    print("（注意：这会处理所有状态的数据）\n")
    
    try:
        all_data = process_all_subjects(
            data_dir=data_dir,
            subject_ids=subject_ids,
            save_output=False  # 设置为True以保存结果
        )
        
        print(f"\n✓ 成功处理 {len(all_data)} 个参与者")
        
        # 显示统计信息
        for subject_id, data in all_data.items():
            print(f"\n参与者 {subject_id}:")
            for state, signal in data.items():
                print(f"  {state:10s}: {signal.shape}")
        
    except Exception as e:
        print(f"✗ 批量处理出错: {e}")
    
    print()


def example_4_verify_specific_channel():
    """示例4：验证特定通道"""
    print("=" * 60)
    print("示例4：验证特定通道的转换")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(123)
    signal_63ch = np.random.randn(63, 500).astype(np.float64)
    
    # 转换
    signal_64ch = num_ch_corr(signal_63ch)
    
    # 验证前23个通道
    print("验证前23个通道（应该完全一致）:")
    diff_1_23 = np.max(np.abs(signal_64ch[0:23, :] - signal_63ch[0:23, :]))
    print(f"  最大差异: {diff_1_23:.2e}")
    print(f"  {'✓ 通过' if diff_1_23 < 1e-10 else '✗ 失败'}")
    
    # 验证Cz通道
    print("\n验证Cz通道（第24个通道，插值得到）:")
    expected_cz = (signal_63ch[6] + signal_63ch[38] + signal_63ch[27] + 
                   signal_63ch[39] + signal_63ch[56] + signal_63ch[11] + 
                   signal_63ch[52] + signal_63ch[22]) / 8.0
    diff_cz = np.max(np.abs(signal_64ch[23, :] - expected_cz))
    print(f"  最大差异: {diff_cz:.2e}")
    print(f"  {'✓ 通过' if diff_cz < 1e-10 else '✗ 失败'}")
    
    # 验证后40个通道
    print("\n验证后40个通道（原24-63映射到25-64）:")
    diff_25_64 = np.max(np.abs(signal_64ch[24:64, :] - signal_63ch[23:63, :]))
    print(f"  最大差异: {diff_25_64:.2e}")
    print(f"  {'✓ 通过' if diff_25_64 < 1e-10 else '✗ 失败'}")
    
    print()


def example_5_save_converted_data():
    """示例5：保存转换后的数据"""
    print("=" * 60)
    print("示例5：保存转换后的数据")
    print("=" * 60)
    
    import scipy.io as sio
    import os
    
    data_dir = "."
    subject_id = 1
    output_dir = "./output_64ch"
    
    try:
        # 加载和转换数据
        data_63ch = load_creativity_data(data_dir, subject_id)
        data_64ch = convert_all_channels(data_63ch)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备保存的数据（使用原始变量名格式）
        save_dict = {}
        for key, value in data_64ch.items():
            if key in ['RST1', 'RST2']:
                var_name = f'Creativity_{subject_id}_{key}'
            else:
                state, trial = key.split('_')
                var_name = f'Creativity_{subject_id}_{trial}_{state}'
            save_dict[var_name] = value
        
        # 保存为MAT文件
        output_file = os.path.join(output_dir, f'Data_Creativity_Sub_{subject_id}_64ch.mat')
        sio.savemat(output_file, save_dict)
        
        print(f"✓ 数据已保存到: {output_file}")
        print(f"  包含 {len(save_dict)} 个变量")
        
    except Exception as e:
        print(f"✗ 保存数据时出错: {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EEG通道转换使用示例")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example_1_single_signal()
    example_2_load_and_convert()
    example_3_batch_processing()
    example_4_verify_specific_channel()
    example_5_save_converted_data()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60 + "\n")


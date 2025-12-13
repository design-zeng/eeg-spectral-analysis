"""
测试脚本：验证Python实现与MATLAB逻辑的一致性

此脚本用于：
1. 验证转换函数的正确性
2. 对比Python和MATLAB的结果（如果可用）
3. 检查数值精度
"""

import numpy as np
import scipy.io as sio
from eeg_channel_conversion import num_ch_corr, verify_conversion, load_creativity_data, convert_all_channels
import os


def test_basic_conversion():
    """测试基本转换功能"""
    print("=" * 60)
    print("测试1：基本转换功能")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    test_data = np.random.randn(63, 1000).astype(np.float64)
    
    # 执行转换
    result = num_ch_corr(test_data)
    
    # 检查维度
    assert result.shape == (64, 1000), f"维度错误：期望(64, 1000)，得到{result.shape}"
    print("✓ 维度检查通过")
    
    # 检查前23个通道
    assert np.allclose(result[0:23, :], test_data[0:23, :]), "前23个通道不匹配"
    print("✓ 前23个通道检查通过")
    
    # 检查Cz通道
    expected_cz = (test_data[6] + test_data[38] + test_data[27] + test_data[39] + 
                   test_data[56] + test_data[11] + test_data[52] + test_data[22]) / 8.0
    assert np.allclose(result[23, :], expected_cz), "Cz通道不匹配"
    print("✓ Cz通道检查通过")
    
    # 检查后40个通道
    assert np.allclose(result[24:64, :], test_data[23:63, :]), "后40个通道不匹配"
    print("✓ 后40个通道检查通过")
    
    print("\n所有基本测试通过！\n")


def test_verify_function():
    """测试验证函数"""
    print("=" * 60)
    print("测试2：验证函数")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(123)
    test_data = np.random.randn(63, 500).astype(np.float64)
    
    # 执行转换
    converted = num_ch_corr(test_data)
    
    # 使用验证函数
    is_valid, report = verify_conversion(test_data, converted)
    
    assert is_valid, "验证失败"
    print("✓ 验证函数测试通过")
    print(f"  报告摘要：")
    for check_name, check_result in report.items():
        if isinstance(check_result, dict) and '通过' in check_result:
            status = "✓" if check_result['通过'] else "✗"
            print(f"    {status} {check_name}")
    
    print("\n验证函数测试通过！\n")


def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试3：边界情况")
    print("=" * 60)
    
    # 测试1：最小时间点数
    test_data = np.random.randn(63, 1).astype(np.float64)
    result = num_ch_corr(test_data)
    assert result.shape == (64, 1), "最小时间点数测试失败"
    print("✓ 最小时间点数测试通过")
    
    # 测试2：大时间点数
    test_data = np.random.randn(63, 100000).astype(np.float64)
    result = num_ch_corr(test_data)
    assert result.shape == (64, 100000), "大时间点数测试失败"
    print("✓ 大时间点数测试通过")
    
    # 测试3：全零数据
    test_data = np.zeros((63, 100), dtype=np.float64)
    result = num_ch_corr(test_data)
    assert np.allclose(result, np.zeros((64, 100))), "全零数据测试失败"
    print("✓ 全零数据测试通过")
    
    # 测试4：全一数据
    test_data = np.ones((63, 100), dtype=np.float64)
    result = num_ch_corr(test_data)
    assert np.allclose(result, np.ones((64, 100))), "全一数据测试失败"
    print("✓ 全一数据测试通过")
    
    # 测试5：错误输入维度
    try:
        wrong_data = np.random.randn(64, 100)  # 已经是64通道
        num_ch_corr(wrong_data)
        assert False, "应该抛出错误"
    except ValueError as e:
        print(f"✓ 错误输入检测通过: {e}")
    
    print("\n边界情况测试通过！\n")


def test_numerical_precision():
    """测试数值精度"""
    print("=" * 60)
    print("测试4：数值精度")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    test_data = np.random.randn(63, 1000).astype(np.float64)
    
    # 执行转换
    result = num_ch_corr(test_data)
    
    # 验证精度
    is_valid, report = verify_conversion(test_data, result, tolerance=1e-10)
    
    if is_valid:
        print("✓ 数值精度测试通过（容差: 1e-10）")
    else:
        print("✗ 数值精度测试失败")
        print("  详细报告：")
        for check_name, check_result in report.items():
            print(f"    {check_name}: {check_result}")
    
    # 检查最大差异
    max_diff_1_23 = report['前23通道检查']['最大差异']
    max_diff_cz = report['Cz通道检查']['最大差异']
    max_diff_25_64 = report['后40通道检查']['最大差异']
    
    print(f"  前23通道最大差异: {max_diff_1_23:.2e}")
    print(f"  Cz通道最大差异: {max_diff_cz:.2e}")
    print(f"  后40通道最大差异: {max_diff_25_64:.2e}")
    
    print("\n数值精度测试完成！\n")


def test_real_data_if_available():
    """如果真实数据可用，测试真实数据"""
    print("=" * 60)
    print("测试5：真实数据（如果可用）")
    print("=" * 60)
    
    data_directory = "."
    subject_id = 1
    
    try:
        # 尝试加载真实数据
        data_63ch = load_creativity_data(data_directory, subject_id)
        
        print(f"✓ 成功加载参与者 {subject_id} 的数据")
        print(f"  数据状态: {list(data_63ch.keys())}")
        
        # 转换所有状态
        data_64ch = convert_all_channels(data_63ch)
        
        print(f"✓ 成功转换所有状态")
        
        # 验证每个状态
        for state in data_63ch.keys():
            is_valid, report = verify_conversion(data_63ch[state], data_64ch[state])
            status = "✓" if is_valid else "✗"
            print(f"  {status} {state}: {data_63ch[state].shape} -> {data_64ch[state].shape}")
        
        print("\n真实数据测试完成！\n")
        
    except FileNotFoundError:
        print(f"⚠ 数据文件未找到，跳过真实数据测试")
        print(f"  预期路径: {os.path.join(data_directory, 'Creativity_EEG_Dataset', f'Data_Creativity_Sub_{subject_id}.mat')}")
        print()
    except Exception as e:
        print(f"✗ 真实数据测试出错: {e}")
        print()


def compare_with_matlab_result():
    """如果MATLAB结果可用，进行对比"""
    print("=" * 60)
    print("测试6：与MATLAB结果对比（如果可用）")
    print("=" * 60)
    
    # 这里假设有一个MATLAB处理后的结果文件
    matlab_result_file = "./matlab_results/Data_Creativity_Sub_1_64ch.mat"
    
    if not os.path.exists(matlab_result_file):
        print("⚠ MATLAB结果文件未找到，跳过对比测试")
        print(f"  预期路径: {matlab_result_file}")
        print()
        return
    
    try:
        # 加载MATLAB结果
        matlab_data = sio.loadmat(matlab_result_file)
        
        # 加载Python处理的原始数据
        data_directory = "."
        python_data_63ch = load_creativity_data(data_directory, 1)
        python_data_64ch = convert_all_channels(python_data_63ch)
        
        # 对比结果
        print("对比结果：")
        for key in python_data_64ch.keys():
            # 构建MATLAB变量名
            if key in ['RST1', 'RST2']:
                matlab_key = f'Creativity_1_{key}'
            else:
                state, trial = key.split('_')
                matlab_key = f'Creativity_1_{trial}_{state}'
            
            if matlab_key in matlab_data:
                python_result = python_data_64ch[key]
                matlab_result = matlab_data[matlab_key]
                
                # 检查维度
                if python_result.shape != matlab_result.shape:
                    print(f"  ✗ {key}: 维度不匹配")
                    print(f"    Python: {python_result.shape}, MATLAB: {matlab_result.shape}")
                    continue
                
                # 检查数值
                max_diff = np.max(np.abs(python_result - matlab_result))
                is_close = np.allclose(python_result, matlab_result, atol=1e-10)
                
                status = "✓" if is_close else "✗"
                print(f"  {status} {key}: 最大差异 = {max_diff:.2e}")
            else:
                print(f"  ⚠ {key}: MATLAB结果中未找到对应变量")
        
        print("\nMATLAB对比测试完成！\n")
        
    except Exception as e:
        print(f"✗ MATLAB对比测试出错: {e}")
        print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行所有测试")
    print("=" * 60 + "\n")
    
    test_basic_conversion()
    test_verify_function()
    test_edge_cases()
    test_numerical_precision()
    test_real_data_if_available()
    compare_with_matlab_result()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()


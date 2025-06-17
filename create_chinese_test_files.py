#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建中文名称的测试文件
"""
import pandas as pd
import os
from datetime import datetime

def create_chinese_name_files():
    """创建带中文名称的测试文件"""
    
    # 创建测试数据
    test_data = {
        '铺叶工段数据记录_流量（kg/h)': [15450, 15600, 15300, 15500, 15400],
        '二润工段数据记录_润后水份（%）': [18.91, 19.2, 18.5, 18.8, 19.0],
        '二润工段数据记录_润后温度（℃）': [54.5, 55.0, 53.8, 54.2, 54.8],
        '打叶工段数据记录_打前水份（%）': [15.8, 16.1, 15.5, 15.9, 16.0],
        '打叶工段数据记录_大中片率': [78.2, 79.1, 77.8, 78.5, 78.8],
        '打叶工段数据记录_叶含梗率': [1.1, 1.2, 1.0, 1.15, 1.05],
        '打叶工段数据记录_梗含叶率': [0.5, 0.6, 0.4, 0.55, 0.45]
    }
    
    df = pd.DataFrame(test_data)
    
    # 创建中文名称的CSV文件
    csv_files = [
        '5.14宝丰.csv',
        '514——天昌.csv', 
        '测试数据_中文名称.csv',
        '烟草生产数据_2025年.csv'
    ]
    
    for filename in csv_files:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 已创建CSV文件: {filename}")
        print(f"   文件大小: {os.path.getsize(filename)} bytes")
    
    # 创建中文名称的Excel文件
    excel_files = [
        '5.14宝丰.xlsx',
        '514——天昌.xlsx',
        '测试数据_中文名称.xlsx',
        '烟草生产数据_2025年.xlsx'
    ]
    
    for filename in excel_files:
        df.to_excel(filename, index=False)
        print(f"✅ 已创建Excel文件: {filename}")
        print(f"   文件大小: {os.path.getsize(filename)} bytes")
    
    print(f"\n📊 共创建了 {len(csv_files + excel_files)} 个测试文件")
    print("现在可以尝试上传这些文件来测试中文文件名支持")
    
    return csv_files + excel_files

def clean_test_files():
    """清理测试文件"""
    test_patterns = [
        '5.14宝丰.*',
        '514——天昌.*',
        '测试数据_中文名称.*',
        '烟草生产数据_2025年.*'
    ]
    
    import glob
    for pattern in test_patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"🗑️ 已删除: {file}")
            except Exception as e:
                print(f"❌ 删除失败 {file}: {str(e)}")

if __name__ == '__main__':
    print("🚀 开始创建中文名称测试文件...")
    files = create_chinese_name_files()
    
    print("\n📝 测试说明:")
    print("1. 启动Flask应用: python run.py")
    print("2. 在浏览器中访问: http://127.0.0.1:5000")
    print("3. 切换到'数据导入'页面")
    print("4. 逐个上传创建的测试文件")
    print("5. 验证文件名是否正确显示")
    
    input("\n按回车键清理测试文件...")
    clean_test_files()
    print("✅ 清理完成!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件上传功能
"""
import pandas as pd
import os
from datetime import datetime

def create_test_files():
    """创建测试文件"""
    # 创建测试CSV文件
    test_data_csv = {
        '铺叶工段数据记录_流量（kg/h)': [15450, 15600, 15300, 15500, 15400],
        '二润工段数据记录_润后水份（%）': [18.91, 19.2, 18.5, 18.8, 19.0],
        '二润工段数据记录_润后温度（℃）': [54.5, 55.0, 53.8, 54.2, 54.8],
        '打叶工段数据记录_打前水份（%）': [15.8, 16.1, 15.5, 15.9, 16.0],
        '打叶工段数据记录_大中片率': [78.2, 79.1, 77.8, 78.5, 78.8],
        '打叶工段数据记录_叶含梗率': [1.1, 1.2, 1.0, 1.15, 1.05],
        '打叶工段数据记录_梗含叶率': [0.5, 0.6, 0.4, 0.55, 0.45]
    }
    
    df_csv = pd.DataFrame(test_data_csv)
    csv_path = 'test_tobacco_data.csv'
    df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已创建CSV测试文件: {csv_path}")
    
    # 创建测试Excel文件
    excel_path = 'test_tobacco_data.xlsx'
    df_csv.to_excel(excel_path, index=False)
    print(f"✅ 已创建Excel测试文件: {excel_path}")
    
    # 检查文件大小
    csv_size = os.path.getsize(csv_path)
    excel_size = os.path.getsize(excel_path)
    
    print(f"📊 CSV文件大小: {csv_size} bytes ({csv_size/1024:.2f} KB)")
    print(f"📊 Excel文件大小: {excel_size} bytes ({excel_size/1024:.2f} KB)")
    
    return csv_path, excel_path

def test_file_reading():
    """测试文件读取功能"""
    csv_path, excel_path = create_test_files()
    
    try:
        # 测试CSV读取
        print("\n🔍 测试CSV文件读取...")
        df_csv = pd.read_csv(csv_path)
        print(f"CSV行数: {len(df_csv)}, 列数: {len(df_csv.columns)}")
        print("CSV列名:")
        for i, col in enumerate(df_csv.columns):
            print(f"  {i+1}. {col}")
        
        # 测试Excel读取
        print("\n🔍 测试Excel文件读取...")
        df_excel = pd.read_excel(excel_path)
        print(f"Excel行数: {len(df_excel)}, 列数: {len(df_excel.columns)}")
        print("Excel列名:")
        for i, col in enumerate(df_excel.columns):
            print(f"  {i+1}. {col}")
        
        # 验证数据一致性
        if df_csv.equals(df_excel):
            print("✅ CSV和Excel数据完全一致")
        else:
            print("⚠️ CSV和Excel数据存在差异")
        
        print("\n📋 数据预览:")
        print(df_csv.head(3).to_string())
        
    except Exception as e:
        print(f"❌ 文件读取测试失败: {str(e)}")
    
    finally:
        # 清理测试文件
        for file_path in [csv_path, excel_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ 已删除测试文件: {file_path}")

if __name__ == '__main__':
    print("🚀 开始文件上传功能测试...")
    test_file_reading()
    print("\n✅ 测试完成!")

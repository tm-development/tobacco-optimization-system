#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件上传功能
"""
import pandas as pd
import os
from datetime import datetime

def create_test_csv():
    """创建测试CSV文件"""
    data = {
        '铺叶工段数据记录_流量（kg/h)': [15450, 15500, 15400],
        '二润工段数据记录_润后水份（%）': [18.5, 18.7, 18.3],
        '二润工段数据记录_润后温度（℃）': [54.0, 54.5, 53.8],
        '打叶工段数据记录_打前水份（%）': [15.8, 15.9, 15.7]
    }
    
    df = pd.DataFrame(data)
    filename = 'test_tobacco_data.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ 创建测试CSV文件: {filename}")
    return filename

def check_upload_directories():
    """检查上传目录是否存在"""
    upload_dir = 'uploads'
    data_dir = os.path.join(upload_dir, 'data')
    model_dir = os.path.join(upload_dir, 'models')
    
    print("📂 检查上传目录...")
    
    for directory in [upload_dir, data_dir, model_dir]:
        if os.path.exists(directory):
            print(f"✅ {directory} 存在")
        else:
            print(f"❌ {directory} 不存在，正在创建...")
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory} 已创建")

def test_file_operations():
    """测试文件操作"""
    print("🧪 测试文件上传功能...")
    
    # 检查目录
    check_upload_directories()
    
    # 创建测试文件
    test_file = create_test_csv()
    
    # 测试文件读取
    try:
        df = pd.read_csv(test_file)
        print(f"📊 文件读取成功: {len(df)} 行 x {len(df.columns)} 列")
        print("列名:", df.columns.tolist())
        print("前3行数据:")
        print(df.head(3))
        
        # 模拟文件移动到上传目录
        import shutil
        target_path = os.path.join('uploads', 'data', test_file)
        shutil.copy2(test_file, target_path)
        print(f"📁 文件已复制到上传目录: {target_path}")
        
        # 清理临时文件
        os.remove(test_file)
        print(f"🗑️ 清理临时文件: {test_file}")
        
    except Exception as e:
        print(f"❌ 文件操作失败: {str(e)}")

if __name__ == '__main__':
    print("🚀 开始测试文件上传功能...")
    test_file_operations()
    print("\n✅ 测试完成!")
    print("\n📝 手动测试步骤:")
    print("1. 运行: python run.py")
    print("2. 访问: http://127.0.0.1:5000")
    print("3. 切换到'数据导入'页面")
    print("4. 点击'选择文件'按钮")
    print("5. 选择 uploads/data/test_tobacco_data.csv 文件")
    print("6. 观察控制台调试信息和上传结果")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的文件上传测试
"""
import os
import pandas as pd

def create_simple_test_file():
    """创建一个简单的测试文件"""
    
    # 创建测试数据
    test_data = {
        '流量': [15450, 15600, 15300],
        '水份': [18.91, 19.2, 18.5],
        '温度': [54.5, 55.0, 53.8],
        '大中片率': [78.2, 79.1, 77.8]
    }
    
    df = pd.DataFrame(test_data)
    
    # 创建简单文件名的CSV文件
    simple_filename = 'test_data.csv'
    df.to_csv(simple_filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ 已创建测试文件: {simple_filename}")
    print(f"📊 文件大小: {os.path.getsize(simple_filename)} bytes")
    
    # 验证文件内容
    df_read = pd.read_csv(simple_filename)
    print(f"📋 文件内容验证: {len(df_read)}行 x {len(df_read.columns)}列")
    print("📄 列名:", df_read.columns.tolist())
    
    return simple_filename

def cleanup_test_file(filename):
    """清理测试文件"""
    if os.path.exists(filename):
        os.remove(filename)
        print(f"🗑️ 已删除测试文件: {filename}")

if __name__ == '__main__':
    print("🚀 创建简单测试文件...")
    filename = create_simple_test_file()
    
    print("\n📝 测试说明:")
    print("1. 启动Flask应用: python run.py")
    print("2. 在浏览器中访问: http://127.0.0.1:5000")
    print("3. 切换到'数据导入'页面")
    print("4. 上传刚创建的test_data.csv文件")
    print("5. 查看控制台调试信息")
    
    input("\n按回车键清理测试文件...")
    cleanup_test_file(filename)
    print("✅ 清理完成!")

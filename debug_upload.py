#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试文件上传功能
"""
import sys
import os
import json
from datetime import datetime
import pandas as pd

# 添加app目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from models import DataFile, FileManager

def test_datafile_serialization():
    """测试DataFile序列化"""
    print("🧪 测试DataFile序列化...")
    
    # 创建测试文件
    test_data = {'test': [1, 2, 3]}
    df = pd.DataFrame(test_data)
    test_file = 'debug_test.csv'
    df.to_csv(test_file, index=False)
    
    try:
        # 创建DataFile对象
        data_file = DataFile(test_file, test_file)
        print(f"✅ DataFile对象创建成功")
        print(f"   文件名: {data_file.filename}")
        print(f"   上传时间类型: {type(data_file.upload_time)}")
        print(f"   上传时间值: {data_file.upload_time}")
        
        # 测试to_dict转换
        file_dict = data_file.to_dict()
        print(f"✅ to_dict()转换成功")
        print(f"   upload_time类型: {type(file_dict['upload_time'])}")
        print(f"   upload_time值: {file_dict['upload_time']}")
        
        # 测试JSON序列化
        json_str = json.dumps(file_dict)
        print(f"✅ JSON序列化成功")
        print(f"   JSON长度: {len(json_str)}")
        
    except Exception as e:
        print(f"❌ 序列化失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)

def test_filemanager():
    """测试FileManager"""
    print("\n🧪 测试FileManager...")
    
    # 确保uploads目录存在
    upload_dir = 'uploads/data'
    os.makedirs(upload_dir, exist_ok=True)
    
    # 创建测试文件
    test_data = {'test': [1, 2, 3]}
    df = pd.DataFrame(test_data)
    test_file = os.path.join(upload_dir, 'filemanager_test.csv')
    df.to_csv(test_file, index=False)
    
    try:
        # 测试获取所有文件
        files = FileManager.get_all_files(upload_dir)
        print(f"✅ 获取文件列表成功，共{len(files)}个文件")
        
        # 测试序列化所有文件
        files_data = []
        for file in files:
            try:
                file_dict = file.to_dict()
                files_data.append(file_dict)
                print(f"   文件 {file.filename} 序列化成功")
            except Exception as e:
                print(f"   文件 {file.filename} 序列化失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 测试JSON序列化
        json_str = json.dumps(files_data)
        print(f"✅ 文件列表JSON序列化成功，共{len(files_data)}个文件")
        
    except Exception as e:
        print(f"❌ FileManager测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)

def test_response_data():
    """测试响应数据结构"""
    print("\n🧪 测试响应数据结构...")
    
    try:
        response_data = {
            'filename': 'test.csv',
            'display_name': 'test.csv',
            'file_type': 'CSV',
            'file_size': 1024,
            'size_mb': 0.001,
            'upload_time': datetime.now().isoformat(),
        }
        
        print(f"✅ 响应数据创建成功")
        print(f"   upload_time类型: {type(response_data['upload_time'])}")
        print(f"   upload_time值: {response_data['upload_time']}")
        
        # 测试JSON序列化
        json_str = json.dumps(response_data)
        print(f"✅ 响应数据JSON序列化成功")
        
    except Exception as e:
        print(f"❌ 响应数据测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🚀 开始调试文件上传序列化问题...")
    
    test_datafile_serialization()
    test_filemanager()
    test_response_data()
    
    print("\n✅ 调试完成!")

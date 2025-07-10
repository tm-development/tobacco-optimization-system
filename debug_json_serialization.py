#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON序列化问题调试工具
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models import DataFile, FileManager
from datetime import datetime
import json

def debug_file_serialization():
    """调试文件序列化问题"""
    print("🔍 开始调试文件序列化问题...")
    
    # 创建测试文件
    test_filename = "debug_test.csv"
    with open(test_filename, 'w', encoding='utf-8') as f:
        f.write("col1,col2\n1,2\n3,4\n")
    
    try:
        # 测试 DataFile 创建
        print("\n1. 测试 DataFile 创建...")
        data_file = DataFile(test_filename, test_filename)
        print(f"   ✅ DataFile 创建成功")
        print(f"   📄 文件名: {data_file.filename}")
        print(f"   🕒 上传时间: {data_file.upload_time} (类型: {type(data_file.upload_time)})")
        
        # 测试 to_dict() 方法
        print("\n2. 测试 to_dict() 方法...")
        file_dict = data_file.to_dict()
        print(f"   ✅ to_dict() 调用成功")
        print(f"   📋 字典内容:")
        for key, value in file_dict.items():
            print(f"      {key}: {value} (类型: {type(value)})")
        
        # 测试 JSON 序列化
        print("\n3. 测试 JSON 序列化...")
        json_str = json.dumps(file_dict, ensure_ascii=False, indent=2)
        print(f"   ✅ JSON 序列化成功")
        print(f"   📄 JSON 内容:\n{json_str}")
        
        # 测试 FileManager.get_all_files()
        print("\n4. 测试 FileManager.get_all_files()...")
        upload_folder = os.path.dirname(os.path.abspath(test_filename))
        files = FileManager.get_all_files(upload_folder, ['.csv'])
        print(f"   ✅ 找到 {len(files)} 个文件")
        
        for file in files:
            if file.filename == test_filename:
                print(f"   📄 测试文件找到")
                print(f"   🕒 上传时间: {file.upload_time} (类型: {type(file.upload_time)})")
                
                # 测试这个文件的序列化
                try:
                    file_dict = file.to_dict()
                    json_str = json.dumps(file_dict, ensure_ascii=False)
                    print(f"   ✅ 文件序列化成功")
                except Exception as e:
                    print(f"   ❌ 文件序列化失败: {str(e)}")
                    print(f"   🔍 问题字段:")
                    for key, value in file_dict.items():
                        try:
                            json.dumps({key: value})
                        except Exception as field_error:
                            print(f"      ❌ {key}: {value} (类型: {type(value)}) - {str(field_error)}")
                break
        
    except Exception as e:
        print(f"❌ 调试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        if os.path.exists(test_filename):
            os.remove(test_filename)
            print(f"\n🗑️ 清理测试文件: {test_filename}")

if __name__ == '__main__':
    debug_file_serialization()

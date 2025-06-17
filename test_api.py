#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件上传API
"""
import requests
import os
import pandas as pd

def create_test_file():
    """创建测试文件"""
    test_data = {
        '流量': [15450, 15600, 15300],
        '水份': [18.91, 19.2, 18.5],
        '温度': [54.5, 55.0, 53.8]
    }
    
    df = pd.DataFrame(test_data)
    filename = 'api_test.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ 创建测试文件: {filename}")
    print(f"📊 文件大小: {os.path.getsize(filename)} bytes")
    
    return filename

def test_upload_api():
    """测试文件上传API"""
    filename = create_test_file()
    
    try:
        # 准备上传文件
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            
            print("🚀 开始测试文件上传API...")
            print(f"📤 上传文件: {filename}")
            
            # 发送上传请求
            response = requests.post('http://127.0.0.1:5000/api/upload_data', files=files)
            
            print(f"📈 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("📋 响应内容:")
                print(f"  成功: {result.get('success')}")
                print(f"  消息: {result.get('message')}")
                
                if 'data' in result:
                    data = result['data']
                    print(f"  文件名: {data.get('filename')}")
                    print(f"  显示名: {data.get('display_name')}")
                    print(f"  文件类型: {data.get('file_type')}")
                    print(f"  文件大小: {data.get('file_size')} bytes")
                
                if result.get('success'):
                    print("✅ 文件上传成功！")
                else:
                    print("❌ 文件上传失败！")
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(f"📄 响应内容: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保Flask应用正在运行在 http://127.0.0.1:5000")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    
    finally:
        # 清理测试文件
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑️ 已删除测试文件: {filename}")

def test_get_files_api():
    """测试获取文件列表API"""
    try:
        print("\n🚀 测试获取文件列表API...")
        response = requests.get('http://127.0.0.1:5000/api/get_uploaded_files')
        
        print(f"📈 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("📋 文件列表:")
            
            if result.get('success'):
                files = result.get('files', [])
                print(f"📊 共找到 {len(files)} 个文件")
                
                for i, file in enumerate(files, 1):
                    print(f"  {i}. {file.get('display_name')} ({file.get('file_type')})")
                    print(f"     文件名: {file.get('filename')}")
                    print(f"     大小: {file.get('file_size')} bytes")
            else:
                print("❌ 获取文件列表失败")
        else:
            print(f"❌ HTTP错误: {response.status_code}")
    
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == '__main__':
    print("🧪 开始API测试...")
    
    # 测试文件上传
    test_upload_api()
    
    # 测试获取文件列表
    test_get_files_api()
    
    print("\n✅ API测试完成!")

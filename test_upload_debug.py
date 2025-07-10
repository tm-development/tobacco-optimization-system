#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件上传功能 - 调试版本
"""
import pandas as pd
import os
import requests
from datetime import datetime

def create_test_csv():
    """创建测试CSV文件"""
    data = {
        '铺叶工段数据记录_流量（kg/h)': [15450, 15600, 15300, 15800, 15500],
        '二润工段数据记录_润后水份（%）': [18.91, 19.2, 18.5, 19.0, 18.8],
        '二润工段数据记录_润后温度（℃）': [54.5, 55.0, 54.0, 55.5, 54.8],
        '打叶工段数据记录_打前水份（%）': [15.8, 16.0, 15.5, 16.2, 15.9]
    }
    
    df = pd.DataFrame(data)
    filename = 'test_upload_debug.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ 创建测试文件: {filename}")
    return filename

def test_upload():
    """测试文件上传"""
    print("🚀 开始测试文件上传功能...")
    
    # 创建测试文件
    filename = create_test_csv()
    
    if not os.path.exists(filename):
        print(f"❌ 测试文件不存在: {filename}")
        return
    
    try:
        # 测试上传
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            
            print(f"📤 上传文件: {filename}")
            print(f"📊 文件大小: {os.path.getsize(filename)} bytes")
            
            # 发送上传请求
            response = requests.post('http://127.0.0.1:5000/api/upload_data', files=files)
            
            print(f"📈 响应状态码: {response.status_code}")
            print(f"📋 响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print("📋 响应内容:")
                    print(f"  成功: {result.get('success')}")
                    print(f"  消息: {result.get('message')}")
                    
                    if 'data' in result:
                        data = result['data']
                        print(f"  文件名: {data.get('filename')}")
                        print(f"  文件类型: {data.get('file_type')}")
                        print(f"  文件大小: {data.get('file_size')} bytes")
                        print(f"  上传时间: {data.get('upload_time')}")
                        
                        if 'rows' in data:
                            print(f"  数据行数: {data.get('rows')}")
                            print(f"  数据列数: {data.get('columns')}")
                    
                    if result.get('success'):
                        print("✅ 文件上传成功！")
                    else:
                        print("❌ 文件上传失败！")
                        
                except Exception as e:
                    print(f"❌ 解析响应JSON失败: {str(e)}")
                    print(f"📄 原始响应内容: {response.text}")
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(f"📄 响应内容: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保Flask应用正在运行在 http://127.0.0.1:5000")
        print("💡 请先运行: python run.py")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑️ 已删除测试文件: {filename}")

def test_config():
    """测试配置检查"""
    print("\n🔧 测试配置检查...")
    
    try:
        response = requests.get('http://127.0.0.1:5000/')
        if response.status_code == 200:
            print("✅ Flask应用运行正常")
        else:
            print(f"❌ Flask应用响应异常: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Flask应用，请先启动应用")
        return False
    except Exception as e:
        print(f"❌ 配置检查失败: {str(e)}")
        return False
    
    return True

if __name__ == '__main__':
    print("🧪 开始文件上传调试测试...")
    
    # 先检查配置
    if test_config():
        # 然后测试上传
        test_upload()
    
    print("\n✅ 调试测试完成!")

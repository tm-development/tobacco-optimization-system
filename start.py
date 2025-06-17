#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
烟草生产优化系统启动脚本
"""
import os
import sys
import subprocess

def check_dependencies():
    """检查依赖包是否已安装"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'openpyxl', 'psutil', 'flask_wtf'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    return missing_packages

def install_dependencies():
    """安装缺失的依赖包"""
    print("正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', '-r', 'requirements.txt'])
        print("依赖包安装完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖包安装失败: {e}")
        return False

def start_app():
    """启动Flask应用"""
    print("启动烟草生产优化系统...")
    print("系统将在 http://127.0.0.1:5000 启动")
    print("按 Ctrl+C 停止服务器")
    
    try:
        from run import app
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"启动失败: {e}")
        return False

def main():
    print("="*60)
    print("           烟草生产优化系统启动器")
    print("="*60)
    
    # 检查依赖
    missing = check_dependencies()
    
    if missing:
        print(f"\n发现缺失依赖: {', '.join(missing)}")
        if input("是否自动安装? (y/n): ").lower() == 'y':
            if not install_dependencies():
                print("依赖安装失败，请手动安装后重试")
                return
        else:
            print("请先安装缺失的依赖包")
            return
    
    print("\n" + "="*60)
    
    # 启动应用
    start_app()

if __name__ == '__main__':
    main()

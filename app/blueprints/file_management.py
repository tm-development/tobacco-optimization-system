from flask import Blueprint, render_template, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
from ..models import DataFile, FileManager
from ..utils import allowed_file
from .user_management import admin_required, token_required

file_management = Blueprint('file_management', __name__)

@file_management.route('/api/upload_data', methods=['POST'])
@admin_required
def upload_data():
    """数据文件上传接口 - 保留原始文件名"""
    try:
        print("=== 文件上传开始 ===")
        print(f"请求方法: {request.method}")
        print(f"请求内容类型: {request.content_type}")
        
        # 检查是否有文件
        if 'file' not in request.files:
            print("❌ 没有文件在请求中")
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        file = request.files['file']
        print(f"📁 接收到文件: {file.filename}")
        
        if file.filename == '':
            print("❌ 文件名为空")
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        # 获取原始文件名
        original_filename = file.filename
        print(f"💾 原始文件名: {original_filename}")
        
        # 检查配置是否存在
        if 'ALLOWED_DATA_EXTENSIONS' not in current_app.config:
            print("❌ 缺少ALLOWED_DATA_EXTENSIONS配置")
            return jsonify({'success': False, 'message': '服务器配置错误'})
        
        # 检查文件扩展名是否允许
        if not allowed_file(original_filename, current_app.config['ALLOWED_DATA_EXTENSIONS']):
            print(f"❌ 不支持的文件类型: {original_filename}")
            return jsonify({'success': False, 'message': '不支持的文件格式，仅支持CSV、Excel格式'})
        
        # 确保上传目录存在
        upload_folder = current_app.config['UPLOAD_FOLDER']
        print(f"📂 配置的上传目录: {upload_folder}")
        print(f"📂 上传目录是否存在: {os.path.exists(upload_folder)}")
        os.makedirs(upload_folder, exist_ok=True)
        print(f"📂 创建目录后是否存在: {os.path.exists(upload_folder)}")
        
        # 检查目录权限
        try:
            test_file = os.path.join(upload_folder, 'test_permission.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print("✅ 目录写入权限正常")
        except Exception as e:
            print(f"❌ 目录权限检查失败: {str(e)}")
            return jsonify({'success': False, 'message': f'目录权限问题: {str(e)}'})
        
        # 保持原始文件名，只对不安全字符进行最小处理
        unsafe_chars = ['/', '\\', '<', '>', ':', '"', '|', '?', '*']
        filename = original_filename
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        print(f"🔧 处理后文件名: {filename}")
        
        # 构建完整文件路径
        file_path = os.path.join(upload_folder, filename)
        print(f"🎯 保存路径: {file_path}")
        
        # 检查文件是否已存在，如果存在则添加序号
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(file_path):
            new_filename = f"{base_name}({counter}){ext}"
            file_path = os.path.join(upload_folder, new_filename)
            filename = new_filename
            counter += 1
            print(f"🔄 文件已存在，使用新名称: {filename}")
        
        # 保存文件
        print("💾 开始保存文件...")
        file.save(file_path)
        print(f"✅ 文件保存完成: {file_path}")
        
        # 验证文件是否真的保存了
        if not os.path.exists(file_path):
            print(f"❌ 文件保存失败，路径不存在: {file_path}")
            return jsonify({'success': False, 'message': '文件保存失败'})
        
        file_size = os.path.getsize(file_path)
        print(f"📊 文件大小: {file_size} bytes")
        
        if file_size == 0:
            print("❌ 文件大小为0，保存失败")
            os.remove(file_path)
            return jsonify({'success': False, 'message': '文件保存失败，文件为空'})
        
        # 获取文件基本信息
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'unknown'
        print(f"📄 文件扩展名: {file_extension}")
        
        # 构建响应数据
        response_data = {
            'filename': filename,
            'display_name': filename,  # 显示名称与文件名一致
            'file_type': file_extension.upper(),
            'file_size': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2),
            'upload_time': datetime.now().isoformat(),
        }
        
        # 尝试读取文件内容信息（可选，失败不影响上传成功）
        try:
            print("📖 开始读取文件内容...")
            if file_extension == 'csv':
                df = pd.read_csv(file_path, nrows=5, encoding='utf-8-sig')
                full_df = pd.read_csv(file_path, encoding='utf-8-sig')
                print("✅ CSV文件读取成功")
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, nrows=5)
                full_df = pd.read_excel(file_path)
                print("✅ Excel文件读取成功")
            else:
                raise ValueError("不支持的文件格式")
            
            response_data.update({
                'rows': len(full_df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'preview': df.fillna('').to_dict('records')
            })
            print(f"📊 数据预览生成成功: {len(full_df)}行 x {len(df.columns)}列")
            
        except Exception as e:
            print(f"⚠️ 文件读取失败，但文件已成功上传: {str(e)}")
            # 即使读取失败，文件上传也是成功的
        
        print("=== 文件上传完成 ===")
        return jsonify({
            'success': True,
            'message': f'文件 "{filename}" 上传成功',
            'data': response_data
        })
        
    except Exception as e:
        print(f"❌ 文件上传异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})

@file_management.route('/api/get_uploaded_files', methods=['GET'])
@token_required
@admin_required
def get_uploaded_files():
    """获取已上传的文件列表"""
    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        files = FileManager.get_all_files(upload_folder)
        
        # 转换为字典列表，确保所有datetime对象都被正确序列化
        files_data = []
        for file in files:
            file_dict = file.to_dict()
            files_data.append(file_dict)
        
        return jsonify({
            'success': True,
            'files': files_data
        })
    except Exception as e:
        print(f"获取文件列表错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取文件列表失败: {str(e)}'})

@file_management.route('/api/search_file', methods=['POST'])
@admin_required
def search_file():
    """搜索文件"""
    try:
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        
        if not keyword:
            return jsonify({'success': False, 'message': '搜索关键词不能为空'})
        
        upload_folder = current_app.config['UPLOAD_FOLDER']
        all_files = FileManager.get_all_files(upload_folder)
        
        # 搜索文件
        matched_files = []
        for file in all_files:
            if keyword.lower() in file.filename.lower():
                matched_files.append(file.to_dict())
        
        return jsonify({
            'success': True,
            'files': matched_files,
            'total': len(matched_files)
        })
        
    except Exception as e:
        print(f"搜索文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'搜索失败: {str(e)}'})

@file_management.route('/api/load_data_file', methods=['POST'])
@admin_required
def load_data_file():
    """加载数据文件并返回预览"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'message': '文件名不能为空'})
        
        upload_folder = current_app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_folder, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '文件不存在'})
        
        # 读取文件
        try:
            file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            if file_extension == 'csv':
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                return jsonify({'success': False, 'message': '不支持的文件格式'})
            
            # 获取文件统计信息
            file_stats = os.stat(file_path)
            file_info = {
                'filename': filename,
                'size': file_stats.st_size,
                'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'preview': df.head(10).fillna('').to_dict('records')
            }
            
            return jsonify({
                'success': True,
                'message': f'文件 "{filename}" 加载成功',
                'data': file_info
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'文件读取失败: {str(e)}'})
        
    except Exception as e:
        print(f"加载数据文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'加载失败: {str(e)}'})

@file_management.route('/api/preview_file/<filename>')
@admin_required
def preview_file(filename):
    """预览文件内容"""
    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_folder, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '文件不存在'})
        
        # 读取文件
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if file_extension == 'csv':
            df = pd.read_csv(file_path, nrows=100, encoding='utf-8-sig')
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file_path, nrows=100)
        else:
            return jsonify({'success': False, 'message': '不支持的文件格式'})
        
        return jsonify({
            'success': True,
            'data': {
                'columns': df.columns.tolist(),
                'preview': df.fillna('').to_dict('records')
            }
        })
        
    except Exception as e:
        print(f"预览文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'预览失败: {str(e)}'})

@file_management.route('/api/delete_file', methods=['DELETE'])
@admin_required
def delete_file():
    """删除文件"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'message': '文件名不能为空'})
        
        upload_folder = current_app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_folder, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '文件不存在'})
        
        # 安全检查：确保文件在上传目录内
        abs_upload_folder = os.path.abspath(upload_folder)
        abs_file_path = os.path.abspath(file_path)
        
        if not abs_file_path.startswith(abs_upload_folder):
            return jsonify({'success': False, 'message': '非法的文件路径'})
        
        try:
            os.remove(file_path)
            print(f"文件删除成功: {filename}")
            return jsonify({
                'success': True,
                'message': f'文件 "{filename}" 删除成功'
            })
        except OSError as e:
            print(f"文件删除失败: {str(e)}")
            return jsonify({'success': False, 'message': f'删除失败: {str(e)}'})
        
    except Exception as e:
        print(f"删除文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'删除失败: {str(e)}'})

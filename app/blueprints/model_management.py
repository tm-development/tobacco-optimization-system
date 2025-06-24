from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import os
import pickle
from datetime import datetime
from ..models import ModelManager
from ..utils import allowed_file
from .user_management import admin_required

model_management = Blueprint('model_management', __name__)

@model_management.route('/api/get_uploaded_models', methods=['GET'])
@admin_required
def get_uploaded_models():
    """获取已上传的模型列表"""
    try:
        model_folder = current_app.config['MODEL_FOLDER']
        models = ModelManager.get_all_models(model_folder)
        
        # 转换为字典列表
        models_data = []
        for model in models:
            model_dict = model.to_dict()
            models_data.append(model_dict)
        
        return jsonify({
            'success': True,
            'models': models_data
        })
    except Exception as e:
        print(f"获取模型列表错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取模型列表失败: {str(e)}'})

@model_management.route('/api/models', methods=['GET'])
@admin_required
def get_models():
    """获取模型列表和状态"""
    try:
        model_folder = current_app.config.get('MODEL_FOLDER', 'models')
        
        # 确保模型目录存在
        os.makedirs(model_folder, exist_ok=True)
        
        models = []
        active_model = None
        
        # 检查激活模型状态文件
        active_model_file = os.path.join(model_folder, 'active_model.txt')
        if os.path.exists(active_model_file):
            with open(active_model_file, 'r', encoding='utf-8') as f:
                active_model = f.read().strip()
        
        # 扫描模型文件
        for filename in os.listdir(model_folder):
            if filename.endswith('.pkl') or filename.endswith('.pickle'):
                file_path = os.path.join(model_folder, filename)
                file_stats = os.stat(file_path)
                
                model_info = {
                    'filename': filename,
                    'display_name': filename.replace('.pkl', '').replace('.pickle', ''),
                    'size': file_stats.st_size,
                    'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                    'upload_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'is_active': filename == active_model,
                    'status': 'active' if filename == active_model else 'inactive'
                }
                
                # 尝试验证模型文件
                try:
                    with open(file_path, 'rb') as f:
                        pickle.load(f)
                    model_info['is_valid'] = True
                except:
                    model_info['is_valid'] = False
                    model_info['status'] = 'invalid'
                
                models.append(model_info)
        
        # 按上传时间排序
        models.sort(key=lambda x: x['upload_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'models': models,
            'active_model': active_model
        })
        
    except Exception as e:
        print(f"获取模型列表错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取模型列表失败: {str(e)}'})

@model_management.route('/api/models/upload', methods=['POST'])
@admin_required
def upload_model():
    """上传模型文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        # 检查文件扩展名
        allowed_extensions = current_app.config.get('ALLOWED_MODEL_EXTENSIONS', {'pkl', 'pickle'})
        if not allowed_file(file.filename, allowed_extensions):
            return jsonify({'success': False, 'message': '不支持的文件格式，仅支持pkl、pickle格式'})
        
        # 获取安全的文件名
        filename = secure_filename(file.filename)
        model_folder = current_app.config['MODEL_FOLDER']
        
        # 确保模型目录存在
        os.makedirs(model_folder, exist_ok=True)
        
        file_path = os.path.join(model_folder, filename)
        
        # 检查文件是否已存在
        if os.path.exists(file_path):
            base_name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(file_path):
                new_filename = f"{base_name}({counter}){ext}"
                file_path = os.path.join(model_folder, new_filename)
                filename = new_filename
                counter += 1
        
        # 保存文件
        file.save(file_path)
        
        # 验证模型文件
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            is_valid = True
        except Exception as e:
            is_valid = False
            print(f"模型验证失败: {str(e)}")
        
        file_stats = os.stat(file_path)
        
        return jsonify({
            'success': True,
            'message': f'模型 "{filename}" 上传成功',
            'model': {
                'filename': filename,
                'size': file_stats.st_size,
                'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                'upload_time': datetime.now().isoformat(),
                'is_valid': is_valid
            }
        })
        
    except Exception as e:
        print(f"模型上传错误: {str(e)}")
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})

@model_management.route('/api/models/<filename>/activate', methods=['POST'])
@admin_required
def activate_model(filename):
    """激活指定的模型"""
    try:
        model_folder = current_app.config['MODEL_FOLDER']
        model_path = os.path.join(model_folder, filename)
        
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'message': '模型文件不存在'})
        
        # 验证模型文件
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            return jsonify({'success': False, 'message': f'模型文件无效: {str(e)}'})
        
        # 记录激活的模型
        active_model_file = os.path.join(model_folder, 'active_model.txt')
        with open(active_model_file, 'w', encoding='utf-8') as f:
            f.write(filename)
        
        print(f"激活模型: {filename}")
        
        return jsonify({
            'success': True,
            'message': f'模型 "{filename}" 激活成功',
            'active_model': filename
        })
        
    except Exception as e:
        print(f"激活模型错误: {str(e)}")
        return jsonify({'success': False, 'message': f'激活失败: {str(e)}'})

@model_management.route('/api/models/<filename>', methods=['DELETE'])
@admin_required
def delete_model(filename):
    """删除指定的模型"""
    try:
        model_folder = current_app.config['MODEL_FOLDER']
        model_path = os.path.join(model_folder, filename)
        
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'message': '模型文件不存在'})
        
        # 安全检查：确保文件在模型目录内
        abs_model_folder = os.path.abspath(model_folder)
        abs_model_path = os.path.abspath(model_path)
        
        if not abs_model_path.startswith(abs_model_folder):
            return jsonify({'success': False, 'message': '非法的文件路径'})
        
        # 检查是否是当前激活的模型
        active_model_file = os.path.join(model_folder, 'active_model.txt')
        current_active = None
        if os.path.exists(active_model_file):
            with open(active_model_file, 'r', encoding='utf-8') as f:
                current_active = f.read().strip()
        
        if current_active == filename:
            # 如果删除的是激活模型，清除激活状态
            os.remove(active_model_file)
            print(f"清除激活模型状态: {filename}")
        
        # 删除模型文件
        os.remove(model_path)
        print(f"模型删除成功: {filename}")
        
        return jsonify({
            'success': True,
            'message': f'模型 "{filename}" 删除成功'
        })
        
    except Exception as e:
        print(f"删除模型错误: {str(e)}")
        return jsonify({'success': False, 'message': f'删除失败: {str(e)}'})

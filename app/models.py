from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# 初始化SQLAlchemy
db = SQLAlchemy()

try:
    import psutil
except ImportError:
    psutil = None

class OptimizationResult(db.Model):
    """优化结果数据库模型"""
    __tablename__ = 'optimization_results'
    
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    input_parameters = db.Column(db.Text, nullable=True)  # 手动输入的参数
    parameters = db.Column(db.Text, nullable=False)  # 原始优化结果参数
    modified_parameters = db.Column(db.Text, nullable=True)  # 修改后的参数
    metrics = db.Column(db.Text, nullable=False)     # JSON字符串格式（包含原始指标）
    modified_metrics = db.Column(db.Text, nullable=True)  # 修改后的指标
    file_path = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    is_modified = db.Column(db.Boolean, default=False)

class OptimizationTask:
    """优化任务模型"""
    
    def __init__(self, task_id, model_name, algorithm, objective, parameters):
        self.task_id = task_id
        self.model_name = model_name
        self.algorithm = algorithm
        self.objective = objective
        self.parameters = parameters
        self.created_at = datetime.now()
        self.status = 'pending'
        self.progress = 0
        self.result = None
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'model_name': self.model_name,
            'algorithm': self.algorithm,
            'objective': self.objective,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'progress': self.progress,
            'result': self.result
        }

class SystemStatus:
    """系统状态模型"""
    
    @staticmethod
    def get_current_status():
        """获取当前系统状态"""
        try:
            if psutil:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                return {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'disk_usage': disk.percent,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'cpu_usage': 0,
                    'memory_usage': 0,
                    'disk_usage': 0,
                    'timestamp': datetime.now().isoformat(),
                    'error': 'psutil not available'
                }
        except Exception as e:
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

class DataFile:
    """数据文件模型"""
    
    def __init__(self, filename, file_path, upload_time=None):
        self.filename = filename
        self.file_path = file_path
        
        # 确保upload_time是datetime对象并立即转换为字符串
        try:
            if upload_time is None:
                upload_time_dt = datetime.fromtimestamp(os.path.getmtime(file_path)) if os.path.exists(file_path) else datetime.now()
            elif isinstance(upload_time, (int, float)):
                upload_time_dt = datetime.fromtimestamp(upload_time)
            elif isinstance(upload_time, datetime):
                upload_time_dt = upload_time
            else:
                # 如果是其他类型，使用当前时间
                upload_time_dt = datetime.now()
            
            # 立即转换为字符串以避免序列化问题
            self.upload_time = upload_time_dt.isoformat()
        except Exception as e:
            print(f"处理upload_time时出错: {str(e)}, 使用当前时间")
            self.upload_time = datetime.now().isoformat()
            
        self.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.file_type = self._get_file_type()
        self.display_name = self._get_display_name()
    
    def _get_file_type(self):
        """获取文件类型"""
        if self.filename.endswith('.csv'):
            return 'CSV'
        elif self.filename.endswith(('.xlsx', '.xls')):
            return 'Excel'
        else:
            return 'Unknown'
    
    def _get_display_name(self):
        """获取显示名称（现在就是原始文件名）"""
        # 由于不再添加时间戳前缀，直接返回文件名
        return self.filename
    
    def to_dict(self):
        return {
            'filename': self.filename,
            'display_name': self.display_name,
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'upload_time': self.upload_time,  # 已经是字符串格式
            'size_mb': round(self.file_size / (1024 * 1024), 2)  # 添加MB大小显示
        }

class ModelFile:
    """模型文件模型"""
    
    def __init__(self, filename, file_path, model_type='unknown', accuracy=None, upload_time=None):
        self.filename = filename
        self.file_path = file_path
        self.model_type = model_type
        self.accuracy = accuracy
        
        # 确保upload_time是datetime对象并转换为字符串
        try:
            if upload_time is None:
                upload_time_dt = datetime.fromtimestamp(os.path.getmtime(file_path)) if os.path.exists(file_path) else datetime.now()
            elif isinstance(upload_time, (int, float)):
                upload_time_dt = datetime.fromtimestamp(upload_time)
            elif isinstance(upload_time, datetime):
                upload_time_dt = upload_time
            else:
                # 如果是其他类型，使用当前时间
                upload_time_dt = datetime.now()
            
            # 立即转换为字符串以避免序列化问题
            self.upload_time = upload_time_dt.isoformat()
        except Exception as e:
            print(f"处理upload_time时出错: {str(e)}, 使用当前时间")
            self.upload_time = datetime.now().isoformat()
            
        self.is_active = False
        self.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.display_name = self._get_display_name()
    
    def _get_display_name(self):
        """获取显示名称（现在就是原始文件名）"""
        # 由于不再添加时间戳前缀，直接返回文件名
        return self.filename
    
    def to_dict(self):
        return {
            'filename': self.filename,
            'display_name': self.display_name,
            'file_path': self.file_path,
            'model_type': self.model_type,
            'accuracy': self.accuracy,
            'file_size': self.file_size,
            'upload_time': self.upload_time,  # 已经是字符串格式
            'is_active': self.is_active,
            'size_mb': round(self.file_size / (1024 * 1024), 2)
        }

class FileManager:
    """文件管理器"""
    
    @staticmethod
    def search_files(upload_folder, search_name, file_extensions=None):
        """搜索文件"""
        if file_extensions is None:
            file_extensions = ['.csv', '.xlsx', '.xls']
        
        found_files = []
        
        if not os.path.exists(upload_folder):
            return found_files
        
        try:
            for filename in os.listdir(upload_folder):
                if any(filename.endswith(ext) for ext in file_extensions):
                    # 获取显示名称
                    display_name = filename
                    if '_' in filename and len(filename.split('_')[0]) == 15:
                        display_name = '_'.join(filename.split('_')[2:])
                    
                    # 模糊匹配
                    if search_name.lower() in display_name.lower() or search_name.lower() in filename.lower():
                        file_path = os.path.join(upload_folder, filename)
                        try:
                            data_file = DataFile(filename, file_path)
                            found_files.append(data_file)
                        except Exception as e:
                            print(f"处理文件 {filename} 时出错: {str(e)}")
                            continue
        except Exception as e:
            print(f"搜索文件时出错: {str(e)}")
        
        return found_files
    
    @staticmethod
    def get_all_files(upload_folder, file_extensions=None):
        """获取所有文件"""
        if file_extensions is None:
            file_extensions = ['.csv', '.xlsx', '.xls']
        
        files = []
        
        if not os.path.exists(upload_folder):
            return files
        
        try:
            for filename in os.listdir(upload_folder):
                if any(filename.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(upload_folder, filename)
                    try:
                        data_file = DataFile(filename, file_path)
                        files.append(data_file)
                    except Exception as e:
                        print(f"处理文件 {filename} 时出错: {str(e)}")
                        continue
            
            # 按上传时间排序（字符串格式的ISO时间可以直接排序）
            files.sort(key=lambda x: x.upload_time, reverse=True)
        except Exception as e:
            print(f"获取文件列表时出错: {str(e)}")
        
        return files
    
    @staticmethod
    def get_file_by_name(upload_folder, filename):
        """根据文件名获取文件"""
        file_path = os.path.join(upload_folder, filename)
        if os.path.exists(file_path):
            try:
                return DataFile(filename, file_path)
            except Exception as e:
                print(f"获取文件 {filename} 时出错: {str(e)}")
                return None
        return None
    
    @staticmethod
    def delete_file(upload_folder, filename):
        """删除文件"""
        file_path = os.path.join(upload_folder, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                print(f"删除文件 {filename} 时出错: {str(e)}")
                return False
        return False

class ModelManager:
    """模型管理器"""
    
    @staticmethod
    def get_all_models(model_folder):
        """获取所有模型文件"""
        models = []
        
        if not os.path.exists(model_folder):
            print(f"模型文件夹不存在: {model_folder}")
            return models
        
        try:
            files = os.listdir(model_folder)
            print(f"模型文件夹中的文件: {files}")
            
            for filename in files:
                if filename.endswith('.pkl'):
                    file_path = os.path.join(model_folder, filename)
                    try:
                        model_file = ModelFile(filename, file_path)
                        models.append(model_file)
                        print(f"添加模型文件: {filename}")
                    except Exception as e:
                        print(f"处理模型文件 {filename} 时出错: {str(e)}")
                        continue
            
            # 按上传时间排序（字符串格式的ISO时间可以直接排序）
            models.sort(key=lambda x: x.upload_time, reverse=True)
            print(f"总共找到 {len(models)} 个模型文件")
        except Exception as e:
            print(f"获取模型列表时出错: {str(e)}")
        
        return models
    
    @staticmethod
    def get_latest_model(model_folder):
        """获取最新的模型文件"""
        models = ModelManager.get_all_models(model_folder)
        return models[0] if models else None
    
    @staticmethod
    def get_model_by_name(model_folder, filename):
        """根据文件名获取模型"""
        file_path = os.path.join(model_folder, filename)
        if os.path.exists(file_path):
            try:
                return ModelFile(filename, file_path)
            except Exception as e:
                print(f"获取模型 {filename} 时出错: {str(e)}")
                return None
        return None

class PredictionResult:
    """预测结果模型"""
    
    def __init__(self, prediction_value, confidence, model_used, data_source, input_params=None):
        self.prediction_value = prediction_value
        self.confidence = confidence
        self.model_used = model_used
        self.data_source = data_source
        self.input_params = input_params or {}
        self.timestamp = datetime.now()
    
    def to_dict(self):
        return {
            'prediction': self.prediction_value,
            'confidence': self.confidence,
            'model_used': self.model_used,
            'data_source': self.data_source,
            'input_params': self.input_params,
            'timestamp': self.timestamp.isoformat()  # 确保转换为ISO格式字符串
        }




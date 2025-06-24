from pathlib import Path
import os
from datetime import timedelta

class Config:
    # 基本配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'tobacco-optimization-secret-key-2025'
    
    # Session配置
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'data')
    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'models')
    
    # 允许的文件扩展名
    ALLOWED_DATA_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    ALLOWED_MODEL_EXTENSIONS = {'pkl'}
    
    # 打印调试信息
    print(f"📂 配置的上传目录: {UPLOAD_FOLDER}")
    print(f"📂 配置的模型目录: {MODEL_FOLDER}")
    print(f"📋 允许的数据文件扩展名: {ALLOWED_DATA_EXTENSIONS}")
    print(f"📋 允许的模型文件扩展名: {ALLOWED_MODEL_EXTENSIONS}")
    
    # 优化算法配置
    OPTIMIZATION_CONFIG = {
        'PSO': {
            'particles': 30,
            'iterations': 100,
            'w': 0.5,
            'c1': 1.5,
            'c2': 1.5
        },
        'GA': {
            'population': 50,
            'generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        },
        'SA': {
            'initial_temp': 1000,
            'cooling_rate': 0.95,
            'min_temp': 0.01
        },
        'GD': {
            'learning_rate': 0.01,
            'iterations': 1000
        }
    }
    
    # 参数范围配置
    PARAMETER_RANGES = {
        '大中片率': {'min': 75, 'max': 85, 'default': 78},
        '叶含梗率': {'min': 0.5, 'max': 1.5, 'default': 1.1},
        '梗含叶率': {'min': 0.3, 'max': 0.8, 'default': 0.5}
    }

def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions
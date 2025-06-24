from flask import Flask
from config import Config
import os

def create_app():
    # 设置模板和静态文件路径为项目根目录
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    app.config.from_object(Config)
    
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    
    # 注册蓝图
    from app.routes import register_blueprints
    register_blueprints(app)
    
    return app
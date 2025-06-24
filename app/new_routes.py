from flask import Blueprint, render_template

# 创建主蓝图
main = Blueprint('main', __name__)

@main.route('/')
def index():
    """主页面"""
    return render_template('index.html')

# 导入所有蓝图
from .blueprints.file_management import file_management
from .blueprints.model_management import model_management
from .blueprints.metric_prediction import metric_prediction
from .blueprints.parameter_optimization import parameter_optimization

def register_blueprints(app):
    """注册所有蓝图到应用"""
    app.register_blueprint(main)
    app.register_blueprint(file_management)
    app.register_blueprint(model_management)
    app.register_blueprint(metric_prediction)
    app.register_blueprint(parameter_optimization)

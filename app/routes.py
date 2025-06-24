from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session

# 创建主蓝图
main = Blueprint('main', __name__)

@main.route('/')
def index():
    """首页 - 直接显示登录页面"""
    # 如果用户已经登录，直接进入主应用
    if 'token' in session:
        from .blueprints.user_management import verify_token
        payload = verify_token(session.get('token'))
        if payload:
            return redirect(url_for('main.dashboard'))
    
    # 显示登录页面
    return render_template('login.html')

@main.route('/dashboard')
def dashboard():
    """主应用页面 - 需要登录验证"""
    # 检查用户是否已登录
    if 'token' not in session:
        return redirect(url_for('main.index'))
    
    # 验证token是否有效
    from .blueprints.user_management import verify_token
    payload = verify_token(session.get('token'))
    if not payload:
        session.clear()
        return redirect(url_for('main.index'))
    
    # 根据用户角色显示不同页面
    user_role = payload.get('role', 'user')
    if user_role == 'admin':
        return render_template('index.html')
    else:
        # 普通用户只显示参数优化页面
        return render_template('user_dashboard.html')



# 导入所有蓝图
from .blueprints.file_management import file_management
from .blueprints.model_management import model_management
from .blueprints.metric_prediction import metric_prediction
from .blueprints.parameter_optimization import parameter_optimization
from .blueprints.user_management import user_management

def register_blueprints(app):
    """注册所有蓝图到应用"""
    app.register_blueprint(main)
    app.register_blueprint(file_management)
    app.register_blueprint(model_management)
    app.register_blueprint(metric_prediction)
    app.register_blueprint(parameter_optimization)
    app.register_blueprint(user_management)

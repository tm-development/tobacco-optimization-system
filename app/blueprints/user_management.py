"""
用户管理蓝图
实现JWT验证和基于角色的权限控制
"""
from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for, current_app
import jwt
from datetime import datetime, timedelta
from functools import wraps
import hashlib

user_management = Blueprint('user_management', __name__)

# 用户数据（在实际项目中应该使用数据库）
USERS = {
    'root': {
        'password': hashlib.sha256('123456'.encode()).hexdigest(),
        'role': 'admin'
    },
    'user': {
        'password': hashlib.sha256('123456'.encode()).hexdigest(),
        'role': 'user'
    }
}

def verify_password(username, password):
    """验证用户密码"""
    if username in USERS:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return USERS[username]['password'] == password_hash
    return False

def generate_token(username):
    """生成JWT令牌"""
    payload = {
        'username': username,
        'role': USERS[username]['role'],
        'exp': datetime.utcnow() + timedelta(hours=24),  # 24小时过期
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """验证JWT令牌"""
    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """JWT令牌验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 检查session中的token
        token = session.get('token')
        if not token:
            return jsonify({'error': '需要登录', 'code': 401}), 401
        
        payload = verify_token(token)
        if not payload:
            session.pop('token', None)
            session.pop('username', None)
            session.pop('role', None)
            return jsonify({'error': '令牌已过期，请重新登录', 'code': 401}), 401
        
        # 将用户信息添加到request中
        request.current_user = {
            'username': payload['username'],
            'role': payload['role']
        }
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """管理员权限验证装饰器"""
    @wraps(f)
    @token_required
    def decorated_function(*args, **kwargs):
        if request.current_user['role'] != 'admin':
            return jsonify({'error': '需要管理员权限', 'code': 403}), 403
        return f(*args, **kwargs)
    return decorated_function

def user_or_admin_required(f):
    """用户或管理员权限验证装饰器"""
    @wraps(f)
    @token_required
    def decorated_function(*args, **kwargs):
        if request.current_user['role'] not in ['user', 'admin']:
            return jsonify({'error': '访问被拒绝', 'code': 403}), 403
        return f(*args, **kwargs)
    return decorated_function

@user_management.route('/login', methods=['GET'])
def login_page():
    """登录页面 - 重定向到根路径"""
    return redirect(url_for('main.index'))

@user_management.route('/api/login', methods=['POST'])
def login():
    """用户登录API"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空', 'success': False}), 400
        
        if verify_password(username, password):
            token = generate_token(username)
            user_role = USERS[username]['role']
              # 将token和用户信息存储到session中
            session['token'] = token
            session['username'] = username
            session['role'] = user_role
            
            return jsonify({
                'success': True,
                'message': '登录成功',
                'token': token,
                'username': username,
                'role': user_role,
                'redirect_url': url_for('main.dashboard')
            })
        else:
            return jsonify({'error': '用户名或密码错误', 'success': False}), 401
            
    except Exception as e:
        return jsonify({'error': f'登录失败: {str(e)}', 'success': False}), 500

@user_management.route('/api/logout', methods=['POST'])
def logout():
    """用户登出API"""
    session.pop('token', None)
    session.pop('username', None)
    session.pop('role', None)
    return jsonify({'success': True, 'message': '登出成功'})

@user_management.route('/api/user/info', methods=['GET'])
@token_required
def get_user_info():
    """获取当前用户信息"""
    return jsonify({
        'username': request.current_user['username'],
        'role': request.current_user['role'],
        'permissions': get_user_permissions(request.current_user['role'])
    })

@user_management.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_auth():
    """验证用户认证状态"""
    return jsonify({
        'authenticated': True,
        'username': request.current_user['username'],
        'role': request.current_user['role']
    })

def get_user_permissions(role):
    """获取用户权限列表"""
    if role == 'admin':
        return {
            'file_management': True,
            'model_management': True,
            'metric_prediction': True,
            'parameter_optimization': True,
            'user_management': True
        }
    elif role == 'user':
        return {
            'file_management': False,
            'model_management': False,
            'metric_prediction': False,
            'parameter_optimization': True,
            'user_management': False
        }
    else:
        return {
            'file_management': False,
            'model_management': False,
            'metric_prediction': False,
            'parameter_optimization': False,
            'user_management': False
        }

def check_permission(permission_name):
    """检查当前用户是否有指定权限"""
    if 'role' not in session:
        return False
    
    permissions = get_user_permissions(session['role'])
    return permissions.get(permission_name, False)

# 在蓝图上下文中添加权限检查函数
@user_management.app_template_global()
def has_permission(permission_name):
    """模板全局函数：检查权限"""
    return check_permission(permission_name)

@user_management.app_template_global()
def current_user_role():
    """模板全局函数：获取当前用户角色"""
    return session.get('role', '')

@user_management.app_template_global()
def current_username():
    """模板全局函数：获取当前用户名"""
    return session.get('username', '')

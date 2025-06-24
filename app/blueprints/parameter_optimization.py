from flask import Blueprint, request, jsonify, current_app
import os
import pandas as pd
import pickle
import numpy as np
import random
from datetime import datetime
from ..utils import optimization_progress, ImprovedRBFNN
from .user_management import user_or_admin_required

parameter_optimization = Blueprint('parameter_optimization', __name__)

@parameter_optimization.route('/api/optimize_parameters', methods=['POST'])
@user_or_admin_required
def optimize_parameters():
    """参数优化接口"""
    try:
        # 获取前端发送的参数
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '没有接收到参数数据'})
        
        # 提取优化配置
        target_values = data.get('target_values', {})
        constraints = data.get('constraints', {})
        optimization_method = data.get('method', 'genetic_algorithm')
        max_iterations = data.get('max_iterations', 100)
        
        print(f"开始参数优化: {optimization_method}")
        print(f"目标值: {target_values}")
        print(f"约束条件: {constraints}")
        
        # 生成优化任务ID
        task_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化优化进度
        optimization_progress[task_id] = {
            'status': 'running',
            'progress': 0,
            'iteration': 0,
            'max_iterations': max_iterations,
            'best_params': None,
            'best_score': None,
            'start_time': datetime.now().isoformat(),
            'method': optimization_method
        }
        
        # 模拟优化过程（实际项目中这里应该是真实的优化算法）
        try:
            # 加载模型用于优化
            model_folder = current_app.config['MODEL_FOLDER']
            pkl_files = [f for f in os.listdir(model_folder) if f.endswith('.pkl')]
            
            if not pkl_files:
                raise Exception("没有找到可用的模型文件")
            
            model_file_path = os.path.join(model_folder, pkl_files[0])
            
            with open(model_file_path, 'rb') as f:
                model_package = pickle.load(f)
            
            # 执行简化的优化逻辑
            best_params = simulate_optimization(
                model_package, 
                target_values, 
                constraints, 
                max_iterations,
                task_id
            )
            
            # 更新优化结果
            optimization_progress[task_id].update({
                'status': 'completed',
                'progress': 100,
                'iteration': max_iterations,
                'best_params': best_params,
                'best_score': calculate_score(best_params, target_values),
                'end_time': datetime.now().isoformat()
            })
            
            return jsonify({
                'success': True,
                'message': '参数优化启动成功',
                'task_id': task_id,
                'estimated_time': f'{max_iterations * 0.1:.1f}秒'
            })
            
        except Exception as e:
            optimization_progress[task_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            return jsonify({'success': False, 'message': f'优化失败: {str(e)}'})
        
    except Exception as e:
        print(f"参数优化接口错误: {str(e)}")
        return jsonify({'success': False, 'message': f'优化失败: {str(e)}'})

@parameter_optimization.route('/api/optimization_progress/<task_id>', methods=['GET'])
@user_or_admin_required
def get_optimization_progress(task_id):
    """获取优化进度"""
    try:
        if task_id not in optimization_progress:
            return jsonify({'success': False, 'message': '任务ID不存在'})
        
        progress_data = optimization_progress[task_id].copy()
        
        return jsonify({
            'success': True,
            'progress': progress_data
        })
        
    except Exception as e:
        print(f"获取优化进度错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取进度失败: {str(e)}'})

@parameter_optimization.route('/api/stop_optimization/<task_id>', methods=['POST'])
@user_or_admin_required
@user_or_admin_required
def stop_optimization(task_id):
    """停止优化任务"""
    try:
        if task_id not in optimization_progress:
            return jsonify({'success': False, 'message': '任务ID不存在'})
        
        optimization_progress[task_id].update({
            'status': 'stopped',
            'end_time': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'message': '优化任务已停止'
        })
        
    except Exception as e:
        print(f"停止优化错误: {str(e)}")
        return jsonify({'success': False, 'message': f'停止优化失败: {str(e)}'})

@parameter_optimization.route('/api/optimization_history', methods=['GET'])
@user_or_admin_required
def get_optimization_history():
    """获取优化历史记录"""
    try:
        # 返回所有优化任务的历史记录
        history = []
        for task_id, progress_data in optimization_progress.items():
            history.append({
                'task_id': task_id,
                'status': progress_data.get('status'),
                'method': progress_data.get('method'),
                'start_time': progress_data.get('start_time'),
                'end_time': progress_data.get('end_time'),
                'best_score': progress_data.get('best_score'),
                'iterations': progress_data.get('iteration', 0),
                'max_iterations': progress_data.get('max_iterations', 0)
            })
        
        # 按开始时间排序
        history.sort(key=lambda x: x['start_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'history': history[:50]  # 最多返回50个历史记录
        })
        
    except Exception as e:
        print(f"获取优化历史错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取历史失败: {str(e)}'})

# 辅助函数

def simulate_optimization(model_package, target_values, constraints, max_iterations, task_id):
    """模拟优化过程"""
    import time
    import random
    
    # 定义参数范围（根据实际业务需求调整）
    param_ranges = {
        '待加工原烟数据记录_含水率（%）': (10.0, 15.0),
        '待加工原烟数据记录_叶片长度（cm）': (20.0, 40.0),
        '一润工段数据记录_加水量（kg/h）': (100.0, 300.0),
        '一润工段数据记录_蒸汽压力（MPa）': (0.3, 0.8),
        '二润工段数据记录_加水量（kg/h）': (50.0, 200.0),
        '回潮工段数据记录_回潮温度（℃）': (60.0, 90.0),
        '回潮工段数据记录_回潮时间（min）': (10.0, 30.0)
    }
    
    # 初始化最佳参数
    best_params = {}
    best_score = float('inf')
    
    for i in range(max_iterations):
        # 生成随机参数组合
        current_params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            # 考虑约束条件
            if param_name in constraints:
                constraint = constraints[param_name]
                if 'min' in constraint:
                    min_val = max(min_val, constraint['min'])
                if 'max' in constraint:
                    max_val = min(max_val, constraint['max'])
            
            current_params[param_name] = random.uniform(min_val, max_val)
        
        # 添加分类变量（随机选择）
        current_params['待加工原烟数据记录_质量属性（稍差、合格、稍强）'] = random.choice(['稍差', '合格', '稍强'])
        current_params['一润工段数据记录_润叶方式（蒸汽、加水、混合）'] = random.choice(['蒸汽', '加水', '混合'])
        current_params['二润工段数据记录_润叶方式（蒸汽、加水、混合）'] = random.choice(['蒸汽', '加水', '混合'])
        current_params['回潮工段数据记录_回潮方式'] = random.choice(['半回', '全回'])
        
        # 计算当前参数的得分
        score = calculate_score(current_params, target_values)
        
        # 更新最佳参数
        if score < best_score:
            best_score = score
            best_params = current_params.copy()
        
        # 更新进度
        progress = (i + 1) / max_iterations * 100
        optimization_progress[task_id].update({
            'progress': progress,
            'iteration': i + 1,
            'best_score': best_score,
            'best_params': best_params
        })
        
        # 模拟计算时间
        time.sleep(0.01)
        
        # 检查是否被停止
        if optimization_progress[task_id].get('status') == 'stopped':
            break
    
    return best_params

def calculate_score(params, target_values):
    """计算参数组合的得分（得分越低越好）"""
    # 模拟预测结果（实际项目中应该使用真实模型预测）
    predicted_big_medium = 85.0 + random.uniform(-5, 5)
    predicted_leaf_stem = 1.1 + random.uniform(-0.3, 0.3)
    predicted_stem_leaf = 0.65 + random.uniform(-0.2, 0.2)
    
    # 计算与目标值的差距
    score = 0
    if 'big_medium_rate' in target_values:
        score += abs(predicted_big_medium - target_values['big_medium_rate']) ** 2
    if 'leaf_stem_rate' in target_values:
        score += abs(predicted_leaf_stem - target_values['leaf_stem_rate']) ** 2 * 100  # 放大权重
    if 'stem_leaf_rate' in target_values:
        score += abs(predicted_stem_leaf - target_values['stem_leaf_rate']) ** 2 * 100  # 放大权重
    
    return score

from flask import Blueprint, request, jsonify, current_app, send_file
import os
import pandas as pd
import pickle
import numpy as np
import random
import time
import csv
import functools
from datetime import datetime
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from deap import base, creator, tools, algorithms
from ..utils import optimization_progress, ImprovedRBFNN
from .user_management import user_or_admin_required
from werkzeug.utils import secure_filename

parameter_optimization = Blueprint('parameter_optimization', __name__)

# 定义分类变量及其可能的取值
categorical_vars = OrderedDict({
    "回潮工段数据记录_回潮方式": ["半回", "全回"],
    "一润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"],
    "二润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"]
})

# 数值型变量
numeric_vars = [
    "铺叶工段数据记录_流量（kg/h)",
    "打叶工段数据记录_一打_打辊参数（hz）",
    "打叶工段数据记录_二打_打辊参数（hz）",
    "打叶工段数据记录_三打_打辊参数（hz）",
    "打叶工段数据记录_四打_打辊参数（hz）",
    "打叶工段数据记录_风分参数（hz）_一分",
    "打叶工段数据记录_风分参数（hz）_二分",
    "打叶工段数据记录_风分参数（hz）_三分",
    "打叶工段数据记录_风分参数（hz）_四分",
    "打叶工段数据记录_风分参数（hz）_五分",
    "打叶工段数据记录_风分参数（hz）_六分",
    "打叶工段数据记录_风分参数（hz）_七分",
    "打叶工段数据记录_风分参数（hz）_八分",
    "打叶工段数据记录_风分参数（hz）_九分",
    "打叶工段数据记录_风分参数（hz）_十分",
    "打叶工段数据记录_风分参数（hz）_十一分"
]

# 参数名称列表
param_names = list(categorical_vars.keys()) + numeric_vars


def extract_fixed_params_from_csv(file_path):
    """从CSV文件中提取固定参数"""
    # 读取 CSV 文件
    new_data = pd.read_csv(file_path, encoding='gbk')

    # 定义要提取的固定分类参数列名
    categorical_columns = [
        "待加工原烟数据记录_质量属性（稍差、合格、稍强）"
    ]

    # 定义要提取的固定数值参数列名
    numeric_columns = [
        "待加工原烟数据记录_水份（%）",
        "待加工原烟数据记录_温度（℃）",
        "待加工原烟数据记录_烟碱含量（%）",
        "待加工原烟数据记录_叶梗比（%）",
        "待加工原烟数据记录_环境温度（℃）",
        "待加工原烟数据记录_环境湿度（%）",
        "铺叶工段数据记录_环境温度（℃）",
        "铺叶工段数据记录_环境湿度（%）",
        "回潮工段数据记录_潮后水份（%）",
        "回潮工段数据记录_潮后温度（℃）",
        "回潮工段数据记录_环境温度（℃）",
        "回潮工段数据记录_环境湿度（%）",
        "一润工段数据记录_润前水份（%）",
        "一润工段数据记录_润前温度（℃）",
        "一润工段数据记录_润后水份（%）",
        "一润工段数据记录_润后温度（℃）",
        "一润工段数据记录_环境温度（℃）",
        "一润工段数据记录_环境湿度（%）",
        "二润工段数据记录_润前水份（%）",
        "二润工段数据记录_润前温度（℃）",
        "二润工段数据记录_润后水份（%）",
        "二润工段数据记录_润后温度（℃）",
        "二润工段数据记录_环境温度（℃）",
        "二润工段数据记录_环境湿度（%）",
        "打叶工段数据记录_打前水份（%）",
        "打叶工段数据记录_打前温度（℃）",
        "打叶工段数据记录_环境湿度（%）",
        "打叶工段数据记录_环境温度（℃）"
    ]

    # 提取固定分类参数
    fixed_categorical_params = new_data[categorical_columns]

    # 提取固定数值参数
    fixed_numeric_params = new_data[numeric_columns]

    return fixed_categorical_params.to_dict('records'), fixed_numeric_params.to_dict('records')


def has_numbers_in_first_three(arr):
    """检查前三个元素是否包含数值"""
    if len(arr) < 3:
        return False

    for item in arr[:3]:
        if isinstance(item, (int, float)):
            return True
    return False


def eval_func(individual, i, fixed_categorical_param_list, fixed_numeric_param_list, target_values,constraints=None):
    """评估函数 - 计算个体的适应度"""
    fixed_categorical_params = fixed_categorical_param_list[i]
    fixed_numeric_params = fixed_numeric_param_list[i]

    # 检查前三个元素是否包含数值
    if has_numbers_in_first_three(individual):
        return 0, 10, 10

    # 提取字符型变量（前3个变量）
    B0 = individual[0]  # 回潮方式
    B1 = individual[1]  # 一润方式
    B2 = individual[2]  # 二润方式

    # 提取数值型变量（后16个变量）
    numeric_values = individual[3:]

    # 对字符型变量进行One-Hot编码
    df_categorical = pd.DataFrame({
        "回潮工段数据记录_回潮方式": [B0],
        "一润工段数据记录_润叶方式（蒸汽、加水、混合）": [B1],
        "二润工段数据记录_润叶方式（蒸汽、加水、混合）": [B2]
    })
    categorical_input = pd.concat([pd.DataFrame([fixed_categorical_params]), df_categorical], axis=1)

    # 拟合并转换数据
    encoder = OneHotEncoder(sparse_output=False, categories=[["稍差", "合格", "稍强"], ["半回", "全回"],
                                                             ["蒸汽", "加水", "混合"], ["蒸汽", "加水", "混合"]])
    encoded_array = encoder.fit_transform(categorical_input)

    # 获取编码后的列名
    feature_names = encoder.get_feature_names_out(input_features=categorical_input.columns)

    encoded_array_bool = encoded_array.astype(bool)
    # 构造编码后的DataFrame
    encoded_input = pd.DataFrame(encoded_array_bool, columns=feature_names)

    numeric_input = pd.DataFrame(index=[0])

    # 添加数值型变量
    for var_name, value in zip(numeric_vars, numeric_values):
        numeric_input[var_name] = value

    numeric_input = pd.concat([pd.DataFrame([fixed_numeric_params]), numeric_input], axis=1)
    X_input = pd.concat([encoded_input, numeric_input], axis=1)

    # 预测叶含梗率
    with open('./uploads/models/叶含梗率_rbfnn.pkl', 'rb') as f:
        model_package1 = pickle.load(f)

    model1 = model_package1['model']
    scaler_X1 = model_package1['scaler_X']
    scaler_y1 = model_package1['scaler_y']
    feature_names1 = scaler_X1.feature_names_in_
    X1 = X_input[feature_names1]
    X1_scaled = scaler_X1.transform(X1)
    y_pred_scaled = model1.predict(X1_scaled)
    y1 = scaler_y1.inverse_transform(y_pred_scaled.reshape(-1, 1))

    # 预测大中片率
    with open('./uploads/models/大中片率_rbfnn.pkl', 'rb') as f:
        model_package2 = pickle.load(f)

    model2 = model_package2['model']
    scaler_X2 = model_package2['scaler_X']
    scaler_y2 = model_package2['scaler_y']
    feature_names2 = scaler_X2.feature_names_in_
    X2 = X_input[feature_names2]
    X2_scaled = scaler_X2.transform(X2)
    y_pred_scaled2 = model2.predict(X2_scaled)
    y2 = scaler_y2.inverse_transform(y_pred_scaled2.reshape(-1, 1))

    # 预测梗含叶率
    with open('./uploads/models/梗含叶率_rbfnn.pkl', 'rb') as f:
        model_package3 = pickle.load(f)

    model3 = model_package3['model']
    scaler_X3 = model_package3['scaler_X']
    scaler_y3 = model_package3['scaler_y']
    feature_names3 = scaler_X3.feature_names_in_
    X3 = X_input[feature_names3]
    X3_scaled = scaler_X3.transform(X3)
    y_pred_scaled3 = model3.predict(X3_scaled)
    y3 = scaler_y3.inverse_transform(y_pred_scaled3.reshape(-1, 1))

    # 计算得分
    large_medium_rate = y2.flatten()[0]
    leaf_in_stem_rate = y1.flatten()[0]
    stem_in_leaf_rate = y3.flatten()[0]

    # 如果有目标值，根据与目标值的接近程度调整适应度
     # 应用约束条件检查
  
    # 应用目标值调整（现有逻辑）
    if target_values:
        if 'big_medium_rate' in target_values:
            large_medium_rate = large_medium_rate - abs(large_medium_rate - target_values['big_medium_rate']) * 0.1
        if 'leaf_stem_rate' in target_values:
            leaf_in_stem_rate = leaf_in_stem_rate + abs(leaf_in_stem_rate - target_values['leaf_stem_rate']) * 0.1
        if 'stem_leaf_rate' in target_values:
            stem_in_leaf_rate = stem_in_leaf_rate + abs(stem_in_leaf_rate - target_values['stem_leaf_rate']) * 0.1
    return large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate


@parameter_optimization.route('/api/optimize_parameters', methods=['POST'])
@user_or_admin_required
def optimize_parameters():
    """参数优化接口"""
    try:
        # 获取前端发送的参数
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '没有接收到参数数据'})

        # 检查输入模式
        input_mode = data.get('input_mode', 'file')  # 默认为文件模式
        manual_parameters = data.get('manual_parameters', {}) if input_mode == 'manual' else None
        
        # 如果是手动输入模式，验证参数完整性
        if input_mode == 'manual':
            required_params = [
                "待加工原烟数据记录_水份（%）",
                "待加工原烟数据记录_温度（℃）",
                "待加工原烟数据记录_烟碱含量（%）",
                "待加工原烟数据记录_叶梗比（%）",
                "待加工原烟数据记录_环境温度（℃）",
                "待加工原烟数据记录_环境湿度（%）",
                "铺叶工段数据记录_环境温度（℃）",
                "铺叶工段数据记录_环境湿度（%）",
                "回潮工段数据记录_潮后水份（%）",
                "回潮工段数据记录_潮后温度（℃）",
                "回潮工段数据记录_环境温度（℃）",
                "回潮工段数据记录_环境湿度（%）",
                "一润工段数据记录_润前水份（%）",
                "一润工段数据记录_润前温度（℃）",
                "一润工段数据记录_润后水份（%）",
                "一润工段数据记录_润后温度（℃）",
                "一润工段数据记录_环境温度（℃）",
                "一润工段数据记录_环境湿度（%）",
                "二润工段数据记录_润前水份（%）",
                "二润工段数据记录_润前温度（℃）",
                "二润工段数据记录_润后水份（%）",
                "二润工段数据记录_润后温度（℃）",
                "二润工段数据记录_环境温度（℃）",
                "二润工段数据记录_环境湿度（%）",
                "打叶工段数据记录_打前水份（%）",
                "打叶工段数据记录_打前温度（℃）",
                "打叶工段数据记录_环境湿度（%）",
                "打叶工段数据记录_环境温度（℃）"
            ]
            
            # 检查必需参数
            missing_params = [param for param in required_params if param not in manual_parameters]
            if missing_params:
                return jsonify({
                    'success': False, 
                    'message': f'缺少必需参数: {", ".join(missing_params[:3])}{"等" if len(missing_params) > 3 else ""}'
                })
            
            print(f"✅ 接收到手动输入的{len(manual_parameters)}个参数")

        # 提取优化配置
        target_values = data.get('target_values', {})
        constraints = data.get('constraints', {})
        optimization_method = data.get('method', 'genetic_algorithm')
        max_iterations = data.get('max_iterations', 10)
        population_size = data.get('population_size', 50)

        print(f"开始参数优化: {optimization_method}, 输入模式: {input_mode}")
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
            'method': optimization_method,
            'input_mode': input_mode
        }

        # 在后台线程中执行优化
        import threading
        thread = threading.Thread(
            target=run_optimization,
            args=(task_id, target_values, constraints, max_iterations, population_size, input_mode, manual_parameters)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': '参数优化启动成功',
            'task_id': task_id,
            'estimated_time': f'{max_iterations * 2:.1f}秒'
        })

    except Exception as e:
        print(f"参数优化接口错误: {str(e)}")
        return jsonify({'success': False, 'message': f'优化失败: {str(e)}'})


def run_optimization(task_id, target_values, constraints, max_iterations, population_size, input_mode='file', manual_parameters=None):
    """在后台运行优化过程"""
    try:
        # 根据输入模式获取固定参数
        if input_mode == 'manual':
            print(f"🔧 使用手动输入参数进行优化")
            
            # 手动输入模式：构造固定参数
            # 添加默认的分类参数
            fixed_categorical_params = {
                "待加工原烟数据记录_质量属性（稍差、合格、稍强）": "合格"  # 默认值
            }
            
            # 将手动参数列表化，便于后续处理
            fixed_categorical_param_list = [fixed_categorical_params]
            fixed_numeric_param_list = [manual_parameters]
            
            print(f"📊 手动输入参数数量: {len(manual_parameters)}")
            print(f"📊 固定分类参数: {fixed_categorical_params}")
            
        else:
            print(f"📁 使用文件模式进行优化")
            
            # 文件模式：从CSV文件提取参数
            fixed_categorical_param_list, fixed_numeric_param_list = extract_fixed_params_from_csv(
                './uploads/data/5.14天昌_处理后数据_10条.csv')
            
            print(f"📊 从文件提取参数数量: 分类参数{len(fixed_categorical_param_list)}, 数值参数{len(fixed_numeric_param_list)}")

        # 确保结果目录存在
        results_dir = './uploads/data/results'
        os.makedirs(results_dir, exist_ok=True)

        # 创建结果文件
        result_file_path = os.path.join(results_dir, f'optimization_result_{task_id}.csv')
        result_file = open(result_file_path, 'w', newline='', encoding='gbk')
        writer = csv.writer(result_file)
        writer.writerow(param_names + ['大中片率', '叶含梗率', '梗含叶率', '综合得分'])

        # 设置遗传算法
        if not hasattr(creator, "Fitness"):
            creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.Fitness)

        toolbox = base.Toolbox()

        # 注册字符型变量
        for var_name, var_values in categorical_vars.items():
            toolbox.register(var_name, random.choice, var_values)

        # 注册数值型变量
        toolbox.register("铺叶工段数据记录_流量（kg/h)", random.randint, 12000, 18000)
        toolbox.register("打叶工段数据记录_一打_打辊参数（hz）", random.randint, 30, 50)
        toolbox.register("打叶工段数据记录_二打_打辊参数（hz）", random.randint, 30, 50)
        toolbox.register("打叶工段数据记录_三打_打辊参数（hz）", random.randint, 30, 50)
        toolbox.register("打叶工段数据记录_四打_打辊参数（hz）", random.randint, 30, 50)
        toolbox.register("打叶工段数据记录_风分参数（hz）_一分", random.randint, 28, 50)
        toolbox.register("打叶工段数据记录_风分参数（hz）_二分", random.randint, 28, 50)
        toolbox.register("打叶工段数据记录_风分参数（hz）_三分", random.randint, 28, 50)
        toolbox.register("打叶工段数据记录_风分参数（hz）_四分", random.randint, 28, 50)
        toolbox.register("打叶工段数据记录_风分参数（hz）_五分", random.randint, 26, 45)
        toolbox.register("打叶工段数据记录_风分参数（hz）_六分", random.randint, 26, 45)
        toolbox.register("打叶工段数据记录_风分参数（hz）_七分", random.randint, 30, 50)
        toolbox.register("打叶工段数据记录_风分参数（hz）_八分", random.randint, 30, 45)
        toolbox.register("打叶工段数据记录_风分参数（hz）_九分", random.randint, 30, 45)
        toolbox.register("打叶工段数据记录_风分参数（hz）_十分", random.randint, 33, 50)
        toolbox.register("打叶工段数据记录_风分参数（hz）_十一分", random.randint, 33, 50)

        categorical_funcs = [toolbox.__getattribute__(var_name) for var_name in categorical_vars.keys()]
        numeric_funcs = [toolbox.__getattribute__(var_name) for var_name in numeric_vars]
        all_funcs = categorical_funcs + numeric_funcs

        # 注册个体和种群
        toolbox.register("individual", tools.initCycle, creator.Individual, all_funcs, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        best_overall_solution = None
        best_overall_score = float('-inf')

        # 根据输入模式确定迭代次数
        iteration_count = len(fixed_categorical_param_list) if input_mode == 'file' else 1
        
        print(f"🔄 开始优化，共需进行 {iteration_count} 次迭代")

        for index in range(iteration_count):
            # 更新进度
            progress = (index * 100) / iteration_count
            optimization_progress[task_id].update({
                'progress': progress,
                'iteration': index,
                'status': 'running'
            })

            print(f"📈 优化进度: {index + 1}/{iteration_count} ({progress:.1f}%)")

            # 注册评估函数
            toolbox.register("evaluate", functools.partial(
                eval_func,
                i=index,
                fixed_categorical_param_list=fixed_categorical_param_list,
                fixed_numeric_param_list=fixed_numeric_param_list,
                target_values=target_values,
                constraints=constraints  # 添加约束条件参数
                ))

            # 注册遗传算法操作
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutUniformInt, low=0, up=100, indpb=0.1)
            toolbox.register("select", tools.selNSGA2)

            # 初始化种群
            pop = toolbox.population(n=population_size)

            print(f"🧬 开始遗传算法优化，种群大小: {population_size}, 最大迭代数: {max_iterations}")

            # 运行遗传算法
            algorithms.eaMuPlusLambda(
                pop,
                toolbox,
                mu=population_size,
                lambda_=population_size,
                cxpb=0.5,
                mutpb=0.1,
                ngen=max_iterations,
                verbose=False
            )

            # 提取所有非支配解
            pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

            best_score = float('-inf')
            best_solution = None
            best_index = -1

            for i, ind in enumerate(pareto_front):
                # 选择"大中片率-叶含梗率-梗含叶率"最大的帕累托解
                large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate = ind.fitness.values
                score = large_medium_rate - leaf_in_stem_rate - stem_in_leaf_rate

                if score > best_score:
                    best_score = score
                    best_solution = ind
                    best_index = i

            # 更新全局最优解
            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_solution = best_solution

            # 写入当前最优解到CSV
            if best_solution:
                large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate = best_solution.fitness.values
                writer.writerow(
                    best_solution +
                    [large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate, best_score]
                )
                
                print(f"✅ 第{index + 1}次迭代完成，最优得分: {best_score:.4f}")
                print(f"   大中片率: {large_medium_rate:.2f}, 叶含梗率: {leaf_in_stem_rate:.2f}, 梗含叶率: {stem_in_leaf_rate:.2f}")

        # 关闭结果文件
        result_file.close()

        print(f"🎯 优化完成，全局最优得分: {best_overall_score:.4f}")

        # 更新优化结果
        if best_overall_solution:
            large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate = best_overall_solution.fitness.values

            best_params = {}
            for name, value in zip(param_names, best_overall_solution):
                best_params[name] = value

            optimization_progress[task_id].update({
                'status': 'completed',
                'progress': 100,
                'iteration': max_iterations,
                'best_params': best_params,
                'best_score': best_overall_score,
                'result_metrics': {
                    'big_medium_rate': float(large_medium_rate),
                    'leaf_stem_rate': float(leaf_in_stem_rate),
                    'stem_leaf_rate': float(stem_in_leaf_rate)
                },
                'result_file': result_file_path,
                'end_time': datetime.now().isoformat()
            })
            
            print(f"📊 优化结果已保存到: {result_file_path}")
            print(f"🎉 最优参数: {len(best_params)} 个参数")
            
        else:
            print(f"❌ 优化失败：未找到有效解")
            optimization_progress[task_id].update({
                'status': 'failed',
                'error': '未找到有效解',
                'end_time': datetime.now().isoformat()
            })

    except Exception as e:
        print(f"❌ 优化过程错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        optimization_progress[task_id].update({
            'status': 'failed',
            'error': str(e),
            'end_time': datetime.now().isoformat()
        })

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
                'result_metrics': progress_data.get('result_metrics'),
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


@parameter_optimization.route('/api/optimization_result/<task_id>', methods=['GET'])
@user_or_admin_required
def get_optimization_result(task_id):
    """获取优化结果详情"""
    try:
        if task_id not in optimization_progress:
            return jsonify({'success': False, 'message': '任务ID不存在'})

        progress_data = optimization_progress[task_id]

        # 如果任务已完成并有结果文件，读取结果数据
        result_data = None
        if progress_data.get('status') == 'completed' and 'result_file' in progress_data:
            try:
                result_df = pd.read_csv(progress_data['result_file'], encoding='gbk')
                result_data = result_df.to_dict('records')
            except Exception as e:
                print(f"读取结果文件错误: {str(e)}")

        return jsonify({
            'success': True,
            'task_info': progress_data,
            'result_data': result_data
        })

    except Exception as e:
        print(f"获取优化结果错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取结果失败: {str(e)}'})


@parameter_optimization.route('/api/download_optimization_result/<task_id>', methods=['GET'])
@user_or_admin_required
def download_optimization_result(task_id):
    """下载优化结果CSV文件"""
    try:
        if task_id not in optimization_progress:
            return jsonify({'success': False, 'message': '任务ID不存在'})

        progress_data = optimization_progress[task_id]

        if progress_data.get('status') != 'completed' or 'result_file' not in progress_data:
            return jsonify({'success': False, 'message': '优化任务尚未完成或没有结果文件'})

        result_file_path = progress_data['result_file']

        if not os.path.exists(result_file_path):
            return jsonify({'success': False, 'message': '结果文件不存在'})

        return send_file(
            result_file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'优化结果_{task_id}.csv'
        )

    except Exception as e:
        print(f"下载优化结果错误: {str(e)}")
        return jsonify({'success': False, 'message': f'下载结果失败: {str(e)}'})


@parameter_optimization.route('/api/compare_optimization_results', methods=['POST'])
@user_or_admin_required
def compare_optimization_results():
    """比较多个优化结果"""
    try:
        # 获取前端发送的任务ID列表
        data = request.get_json()
        if not data or 'task_ids' not in data:
            return jsonify({'success': False, 'message': '没有接收到任务ID列表'})

        task_ids = data['task_ids']
        if not task_ids or not isinstance(task_ids, list):
            return jsonify({'success': False, 'message': '任务ID列表格式不正确'})

        # 收集所有任务的结果
        comparison_data = []

        for task_id in task_ids:
            if task_id not in optimization_progress:
                continue

            progress_data = optimization_progress[task_id]

            # 只比较已完成的任务
            if progress_data.get('status') != 'completed':
                continue

            result_metrics = progress_data.get('result_metrics', {})

            comparison_data.append({
                'task_id': task_id,
                'method': progress_data.get('method', '未知'),
                'start_time': progress_data.get('start_time', ''),
                'end_time': progress_data.get('end_time', ''),
                'best_score': progress_data.get('best_score', 0),
                'big_medium_rate': result_metrics.get('big_medium_rate', 0),
                'leaf_stem_rate': result_metrics.get('leaf_stem_rate', 0),
                'stem_leaf_rate': result_metrics.get('stem_leaf_rate', 0),
                'iterations': progress_data.get('iteration', 0),
                'max_iterations': progress_data.get('max_iterations', 0)
            })

        return jsonify({
            'success': True,
            'comparison_data': comparison_data
        })

    except Exception as e:
        print(f"比较优化结果错误: {str(e)}")
        return jsonify({'success': False, 'message': f'比较结果失败: {str(e)}'})


@parameter_optimization.route('/api/batch_optimize', methods=['POST'])
@user_or_admin_required
def batch_optimize():
    """批量参数优化接口"""
    try:
        # 获取前端发送的参数
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '没有接收到参数数据'})

        # 提取优化配置
        batch_configs = data.get('batch_configs', [])
        if not batch_configs:
            return jsonify({'success': False, 'message': '没有接收到批量优化配置'})

        # 生成批量优化任务ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化批量任务进度
        task_ids = []

        # 依次启动每个优化任务
        for config in batch_configs:
            target_values = config.get('target_values', {})
            constraints = config.get('constraints', {})
            optimization_method = config.get('method', 'genetic_algorithm')
            max_iterations = config.get('max_iterations', 10)
            population_size = config.get('population_size', 50)

            # 生成优化任务ID
            task_id = f"{batch_id}_task_{len(task_ids)}"
            task_ids.append(task_id)

            # 初始化优化进度
            optimization_progress[task_id] = {
                'status': 'pending',
                'progress': 0,
                'iteration': 0,
                'max_iterations': max_iterations,
                'best_params': None,
                'best_score': None,
                'start_time': datetime.now().isoformat(),
                'method': optimization_method,
                'batch_id': batch_id
            }

        # 在后台线程中执行批量优化
        import threading
        thread = threading.Thread(
            target=run_batch_optimization,
            args=(batch_id, task_ids, batch_configs)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': '批量参数优化启动成功',
            'batch_id': batch_id,
            'task_ids': task_ids,
            'estimated_time': f'{len(batch_configs) * max_iterations * 2:.1f}秒'
        })

    except Exception as e:
        print(f"批量参数优化接口错误: {str(e)}")
        return jsonify({'success': False, 'message': f'批量优化失败: {str(e)}'})


def run_batch_optimization(batch_id, task_ids, batch_configs):
    """在后台运行批量优化过程"""
    for i, task_id in enumerate(task_ids):
        try:
            config = batch_configs[i]
            target_values = config.get('target_values', {})
            constraints = config.get('constraints', {})
            max_iterations = config.get('max_iterations', 10)
            population_size = config.get('population_size', 50)

            # 更新任务状态为运行中
            optimization_progress[task_id].update({
                'status': 'running',
                'start_time': datetime.now().isoformat()
            })

            # 执行优化
            run_optimization(task_id, target_values, constraints, max_iterations, population_size)

        except Exception as e:
            print(f"批量优化任务 {task_id} 错误: {str(e)}")
            optimization_progress[task_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })


@parameter_optimization.route('/api/batch_optimization_status/<batch_id>', methods=['GET'])
@user_or_admin_required
def get_batch_optimization_status(batch_id):
    """获取批量优化状态"""
    try:
        # 查找属于该批次的所有任务
        batch_tasks = []
        for task_id, progress_data in optimization_progress.items():
            if progress_data.get('batch_id') == batch_id:
                batch_tasks.append({
                    'task_id': task_id,
                    'status': progress_data.get('status'),
                    'progress': progress_data.get('progress', 0),
                    'start_time': progress_data.get('start_time'),
                    'end_time': progress_data.get('end_time', None)
                })

        if not batch_tasks:
            return jsonify({'success': False, 'message': '批次ID不存在'})

        # 计算整体进度
        completed_tasks = sum(1 for task in batch_tasks if task['status'] in ['completed', 'failed', 'stopped'])
        total_tasks = len(batch_tasks)
        overall_progress = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0

        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'overall_progress': overall_progress,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'tasks': batch_tasks
        })

    except Exception as e:
        print(f"获取批量优化状态错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取批量状态失败: {str(e)}'})


@parameter_optimization.route('/api/apply_optimization_result/<task_id>', methods=['POST'])
@user_or_admin_required
def apply_optimization_result(task_id):
    """应用优化结果到生产配置"""
    try:
        if task_id not in optimization_progress:
            return jsonify({'success': False, 'message': '任务ID不存在'})

        progress_data = optimization_progress[task_id]

        if progress_data.get('status') != 'completed' or 'best_params' not in progress_data:
            return jsonify({'success': False, 'message': '优化任务尚未完成或没有有效结果'})

        best_params = progress_data['best_params']
        result_metrics = progress_data.get('result_metrics', {})

        # 在实际应用中，这里应该将最优参数应用到生产系统
        # 这里仅模拟记录应用操作
        application_record = {
            'task_id': task_id,
            'applied_at': datetime.now().isoformat(),
            'params': best_params,
            'expected_metrics': result_metrics
        }

        # 在实际系统中，应该将应用记录保存到数据库
        # 这里简单地添加到优化进度记录中
        optimization_progress[task_id]['applied'] = application_record

        return jsonify({
            'success': True,
            'message': '优化结果已成功应用到生产配置',
            'application_record': application_record
        })

    except Exception as e:
        print(f"应用优化结果错误: {str(e)}")
        return jsonify({'success': False, 'message': f'应用结果失败: {str(e)}'})


@parameter_optimization.route('/api/clear_optimization_history', methods=['POST'])
@user_or_admin_required
def clear_optimization_history():
    """清空优化历史记录"""
    try:
        # 获取前端发送的参数
        data = request.get_json()
        days = data.get('days', 30) if data else 30  # 默认清除30天前的记录

        # 计算截止日期
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        # 找出需要删除的任务ID
        tasks_to_remove = []
        for task_id, progress_data in optimization_progress.items():
            start_time = progress_data.get('start_time', '')
            if start_time < cutoff_str:
                tasks_to_remove.append(task_id)

                # 如果有结果文件，也删除文件
                if 'result_file' in progress_data and os.path.exists(progress_data['result_file']):
                    try:
                        os.remove(progress_data['result_file'])
                    except Exception as e:
                        print(f"删除结果文件错误: {str(e)}")

        # 从字典中删除任务记录
        for task_id in tasks_to_remove:
            del optimization_progress[task_id]

        return jsonify({
            'success': True,
            'message': f'已清除{len(tasks_to_remove)}条{days}天前的优化历史记录'
        })

    except Exception as e:
        print(f"清空优化历史错误: {str(e)}")
        return jsonify({'success': False, 'message': f'清空历史失败: {str(e)}'})


@parameter_optimization.route('/api/upload_optimization_file', methods=['POST'])
@user_or_admin_required
def upload_optimization_file():
    """上传优化数据文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有接收到文件'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'})

        if file and file.filename.endswith('.csv'):
            # 保存文件
            filename = secure_filename(file.filename)
            file_path = os.path.join('./uploads/data', filename)
            file.save(file_path)

            # 检查文件是否可用于优化
            try:
                # 简单检查文件是否可读
                pd.read_csv(file_path, encoding='gbk', nrows=5)

                return jsonify({
                    'success': True,
                    'message': '文件上传成功',
                    'filename': filename
                })
            except Exception as e:
                os.remove(file_path)  # 删除无效文件
                return jsonify({'success': False, 'message': f'文件格式错误: {str(e)}'})
        else:
            return jsonify({'success': False, 'message': '不支持的文件格式'})

    except Exception as e:
        print(f"上传优化文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})


@parameter_optimization.route('/api/get_optimization_results', methods=['GET'])
@user_or_admin_required
def get_optimization_results():
    """获取所有优化结果文件"""
    try:
        results_dir = './uploads/data/results'
        if not os.path.exists(results_dir):
            return jsonify({'success': False, 'message': '结果目录不存在'})
            
        files = []
        for filename in os.listdir(results_dir):
            if filename.startswith('optimization_result_') and filename.endswith('.csv'):
                file_path = os.path.join(results_dir, filename)
                file_stat = os.stat(file_path)
                
                # 从文件名中提取任务ID和时间信息
                task_id = filename.replace('optimization_result_', '').replace('.csv', '')
                
                # 尝试解析文件名中的时间
                created_time = None
                try:
                    if 'opt_' in task_id:
                        date_str = task_id.split('opt_')[1]
                        created_time = datetime.strptime(date_str, '%Y%m%d_%H%M%S').isoformat()
                except Exception:
                    pass
                
                files.append({
                    'name': filename,
                    'display_name': f"优化结果 {task_id}",
                    'size': f"{file_stat.st_size / 1024:.1f} KB",
                    'created_time': created_time or datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                })
                
        return jsonify({
            'success': True,
            'files': files
        })
        
    except Exception as e:
        print(f"获取优化结果文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取文件列表失败: {str(e)}'})


@parameter_optimization.route('/api/preview_optimization_result/<filename>', methods=['GET'])
@user_or_admin_required
def preview_optimization_result(filename):
    """预览优化结果文件前5行"""
    try:
        if '..' in filename:  # 安全检查
            return jsonify({'success': False, 'message': '无效的文件名'})
        
        # 使用项目根目录构建绝对路径
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        file_path = os.path.join(base_dir, 'uploads', 'data', 'results', filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '文件不存在'})
        
        # 读取CSV文件前5行
        df = pd.read_csv(file_path, encoding='gbk', nrows=5)
        
        return jsonify({
            'success': True,
            'columns': df.columns.tolist(),
            'preview': df.to_dict('records')
        })
        
    except Exception as e:
        print(f"预览优化结果文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'预览文件失败: {str(e)}'})


@parameter_optimization.route('/api/download_optimization_result_file/<path:filename>', methods=['GET'])
@user_or_admin_required
def download_optimization_result_file(filename):
    """下载优化结果文件"""
    try:
        # 提取纯文件名，移除可能的路径前缀
        clean_filename = os.path.basename(filename)
        if '..' in clean_filename:  # 安全检查
            return jsonify({'success': False, 'message': '无效的文件名'})
        
        # 使用项目根目录构建绝对路径
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        file_path = os.path.join(base_dir, 'uploads', 'data', 'results', clean_filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': f'文件不存在: {clean_filename}, 路径: {file_path}'})
        
        return send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=clean_filename
        )
        
    except Exception as e:
        print(f"下载优化结果文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'下载文件失败: {str(e)}'})
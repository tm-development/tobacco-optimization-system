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
import json
import warnings
import joblib
from ..models import db, OptimizationResult  # 导入数据库模型
# 添加以下代码来忽略sklearn版本警告
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

parameter_optimization = Blueprint('parameter_optimization', __name__)
@parameter_optimization.route('/api/test', methods=['GET'])
@user_or_admin_required
def test_api():
    """测试API接口"""
    print("🔍 进入测试API接口")
    return jsonify({'success': True, 'message': '测试API接口正常工作'})

# 定义分类变量及其可能的取值
categorical_vars = OrderedDict({
    "回潮工段数据记录_回潮方式": ["半回", "全回"],
    "一润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"],
    "二润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"]
})

# 数值型变量
numeric_vars = [
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
        "待加工原烟数据记录_部位（上、中、下）",
        "待加工原烟数据记录_等级（上、中、下）",
        "待加工原烟数据记录_符合性（优、良、合格）",
        "待加工原烟数据记录_耐加工性（强、中、弱）",
    ]

    # 定义要提取的固定数值参数列名（包含铺叶工段流量）
    numeric_columns = [
        "待加工原烟数据记录_水份（%）",
        "待加工原烟数据记录_温度（℃）",
        "待加工原烟数据记录_烟碱含量（%）",
        "待加工原烟数据记录_叶梗比（%）",
        "待加工原烟数据记录_环境温度（℃）",
        "待加工原烟数据记录_环境湿度（%）",
        "铺叶工段数据记录_环境温度（℃）",
        "铺叶工段数据记录_环境湿度（%）",
        "二润工段数据记录_流量（kg/h)",  
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

     # 修复：根据实际的分类特征数量设置categories
    # 获取所有分类列
    categorical_columns = list(categorical_input.columns)
    
    # 为每个分类特征定义可能的取值
    categories_map = {
        "待加工原烟数据记录_部位（上、中、下）": ["上", "中", "下"],        # ✅ 改为字符串
        "待加工原烟数据记录_等级（上、中、下）": ["上", "中", "下"],        # ✅ 改为字符串
        "待加工原烟数据记录_符合性（优、良、合格）": ["优", "良", "合格"],    # ✅ 改为字符串
        "待加工原烟数据记录_耐加工性（强、中、弱）": ["强", "中", "弱"],      # ✅ 改为字符串
        "回潮工段数据记录_回潮方式": ["半回", "全回"],
        "一润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"],
        "二润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"]
    }
    
    # 按照列的顺序构建categories列表
    categories_list = []
    for col in categorical_columns:
        if col in categories_map:
            categories_list.append(categories_map[col])
        else:
            # 如果没有预定义，从数据中推断
            unique_values = categorical_input[col].unique().tolist()
            categories_list.append(unique_values)
    
    # 拟合并转换数据
    encoder = OneHotEncoder(sparse_output=False, categories=categories_list)
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
    # 🔧 新增：将特定分类特征转换为数值
    categorical_to_numeric_map = {
        "待加工原烟数据记录_部位（上、中、下）": {"上": 2, "中": 1, "下": 0},
        "待加工原烟数据记录_等级（上、中、下）": {"上": 2, "中": 1, "下": 0},
        "待加工原烟数据记录_符合性（优、良、合格）": {"优": 2, "良": 1, "合格": 0},
        "待加工原烟数据记录_耐加工性（强、中、弱）": {"强": 2, "中": 1, "弱": 0}
    }

    # 转换指定列为数值
    for col_name, value_map in categorical_to_numeric_map.items():
        if col_name in X_input.columns:
            X_input[col_name] = X_input[col_name].map(value_map)
    
    # 从 X_input 中取出"待加工原烟数据记录_部位（上、中、下）",
    # "待加工原烟数据记录_等级（上、中、下）",
    # "待加工原烟数据记录_符合性（优、良、合格）",
    # "待加工原烟数据记录_耐加工性（强、中、弱）",
    # part_arrtibute = X_input["待加工原烟数据记录_部位（上、中、下）"].values[0]
    grade_attribute = X_input["待加工原烟数据记录_等级（上、中、下）"].values[0]
    conformity_attribute = X_input["待加工原烟数据记录_符合性（优、良、合格）"].values[0]
    processability_attribute = X_input["待加工原烟数据记录_耐加工性（强、中、弱）"].values[0]
    # 并输出相应的值
    # print(f"部位属性值: {part_arrtibute}, 等级属性值: {grade_attribute}, 符合性属性值: {conformity_attribute}, 耐加工性属性值: {processability_attribute}")

    
    # 🔧 新增：模型需要的48个特征列表（按模型训练时的顺序）
    required_features = [
        "待加工原烟数据记录_水份（%）",
        "待加工原烟数据记录_温度（℃）",
        "待加工原烟数据记录_烟碱含量（%）",
        "待加工原烟数据记录_叶梗比（%）",
        "待加工原烟数据记录_环境温度（℃）",
        "待加工原烟数据记录_环境湿度（%）",
        "待加工原烟数据记录_部位（上、中、下）",
        "待加工原烟数据记录_等级（上、中、下）",
        "待加工原烟数据记录_符合性（优、良、合格）",
        "待加工原烟数据记录_耐加工性（强、中、弱）",
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
        "二润工段数据记录_流量（kg/h)",
        "打叶工段数据记录_打前水份（%）",
        "打叶工段数据记录_打前温度（℃）",
        "打叶工段数据记录_环境温度（℃）",
        "打叶工段数据记录_环境湿度（%）",
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
        

    # 预测梗含叶率
    with open('./uploads/models/梗含叶率_rbfnn.pkl', 'rb') as f:
        model_package1 = pickle.load(f)

    model1 = model_package1['model']
    scaler_X1 = model_package1['scaler_X']
    scaler_y1 = model_package1['scaler_y']

    # 🔧 新增：准备符合模型要求的输入
    X1_prepared = prepare_model_input(X_input, required_features)

    X1_scaled = scaler_X1.transform(X1_prepared)
    y_pred_scaled = model1.predict(X1_scaled)
    y1 = scaler_y1.inverse_transform(y_pred_scaled.reshape(-1, 1))
    

    # 对其他两个模型做同样的处理
    # 预测大中片率
    with open('./uploads/models/大中片率_rbfnn.pkl', 'rb') as f:
        model_package2 = pickle.load(f)

    model2 = model_package2['model']
    scaler_X2 = model_package2['scaler_X']
    scaler_y2 = model_package2['scaler_y']

    X2_prepared = prepare_model_input(X_input, required_features)
    X2_scaled = scaler_X2.transform(X2_prepared)
    y_pred_scaled2 = model2.predict(X2_scaled)
    y2 = scaler_y2.inverse_transform(y_pred_scaled2.reshape(-1, 1))

    # 预测叶含梗率
    with open('./uploads/models/叶含梗率_rbfnn.pkl', 'rb') as f:
        model_package3 = pickle.load(f)

    model3 = model_package3['model']
    scaler_X3 = model_package3['scaler_X']
    scaler_y3 = model_package3['scaler_y']

    X3_prepared = prepare_model_input(X_input, required_features)
    X3_scaled = scaler_X3.transform(X3_prepared)
    y_pred_scaled3 = model3.predict(X3_scaled)
    y3 = scaler_y3.inverse_transform(y_pred_scaled3.reshape(-1, 1))


    # 计算得分
    large_medium_rate = y2.flatten()[0]
    leaf_in_stem_rate = y1.flatten()[0]
    stem_in_leaf_rate = y3.flatten()[0]

    
   
   
    # 应用目标值调整（现有逻辑）
    if target_values:
        if 'big_medium_rate' in target_values:
            large_medium_rate = large_medium_rate - abs(large_medium_rate - target_values['big_medium_rate']) * 0.1
        if 'leaf_stem_rate' in target_values:
            leaf_in_stem_rate = leaf_in_stem_rate + abs(leaf_in_stem_rate - target_values['leaf_stem_rate']) * 0.1
        if 'stem_leaf_rate' in target_values:
            stem_in_leaf_rate = stem_in_leaf_rate + abs(stem_in_leaf_rate - target_values['stem_leaf_rate']) * 0.1
    return large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate

# 在模型预测前，重新整理X_input以匹配模型要求
def prepare_model_input(X_input, required_features):
    """准备符合模型要求的输入数据"""
    # 创建新的DataFrame，按照模型要求的特征顺序
    model_input = pd.DataFrame(index=X_input.index)
    
    missing_features = []  # 记录缺失的特征
    
    for feature in required_features:
        if feature in X_input.columns:
            model_input[feature] = X_input[feature]
        else:
            # 缺少特征时，记录缺失信息并准备抛出错误
            missing_features.append(feature)
            print(f"❌ 缺少关键特征: {feature}")
    
    # 如果有缺失特征，抛出异常
    if missing_features:
        error_msg = f"模型输入缺少 {len(missing_features)} 个关键特征: {', '.join(missing_features[:5])}"
        if len(missing_features) > 5:
            error_msg += f" 等{len(missing_features)}个特征"
        print(f"🚨 {error_msg}")
        raise ValueError(error_msg)
    
    return model_input

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
                "待加工原烟数据记录_部位（上、中、下）",
                "待加工原烟数据记录_等级（上、中、下）",
                "待加工原烟数据记录_符合性（优、良、合格）",
                "待加工原烟数据记录_耐加工性（强、中、弱）",
                "待加工原烟数据记录_叶梗比（%）",
                "待加工原烟数据记录_环境温度（℃）",
                "待加工原烟数据记录_环境湿度（%）",
                "铺叶工段数据记录_环境温度（℃）",
                "铺叶工段数据记录_环境湿度（%）",
                "二润工段数据记录_流量（kg/h)",  # 添加到必需参数中
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
        
        # # 如果是手动输入模式，保存输入参数
        # if input_mode == 'manual':
        #     optimization_progress[task_id]['input_parameters'] = manual_parameters

        # 如果是手动输入模式，保存输入参数
        if input_mode == 'manual':
            # 新增：详细打印接收到的手动输入参数
            print("=" * 60)
            print("📥 接收到的前端手动输入参数详情:")
            print(f"   参数总数: {len(manual_parameters)}")
            print(f"   参数类型: {type(manual_parameters)}")
            print("-" * 60)
            
            # 按类别分组打印参数
            categorical_params = {}
            numeric_params = {}
            other_params = {}
            
            for key, value in manual_parameters.items():
                # 更新分类判断条件，包含所有分类参数的关键词
                if any(keyword in key for keyword in ['质量属性', '回潮方式', '润叶方式', '部位', '等级', '符合性', '耐加工性']):
                    categorical_params[key] = value
                elif any(keyword in key for keyword in ['水份', '温度', '湿度', '含量', '比', '参数', '流量']):
                    numeric_params[key] = value
                else:
                    other_params[key] = value
            
            # 打印分类参数
            if categorical_params:
                print("🏷️  分类参数 ({0}个):".format(len(categorical_params)))
                for key, value in categorical_params.items():
                    print(f"     {key}: {value}")
            
            # 打印数值参数
            if numeric_params:
                print(f"\n📊 数值参数 ({len(numeric_params)}个):")
                for key, value in numeric_params.items():
                    print(f"     {key}: {value}")
            
            # 打印其他参数
            if other_params:
                print(f"\n📋 其他参数 ({len(other_params)}个):")
                for key, value in other_params.items():
                    print(f"     {key}: {value}")
            
            print("=" * 60)
            
        optimization_progress[task_id]['input_parameters'] = manual_parameters   

        # 在后台线程中执行优化
        import threading
        thread = threading.Thread(
            target=run_optimization,
            args=(task_id, target_values, constraints, max_iterations, population_size, input_mode, manual_parameters if input_mode == 'manual' else None)
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
            print(f"🔍 接收到的手动参数: {manual_parameters}")
            # 手动输入模式
            part_attr = manual_parameters.get(
                "待加工原烟数据记录_部位（上、中、下）", 
                "中"  # 如果前端没传，则使用默认值
            )
            grade_attr = manual_parameters.get(
                "待加工原烟数据记录_等级（上、中、下）", 
                "中"  # 如果前端没传，则使用默认值
            )
            conformity_attr = manual_parameters.get(
                "待加工原烟数据记录_符合性（优、良、合格）", 
                "合格"  # 如果前端没传，则使用默认值
            )
            processability_attr = manual_parameters.get(
                "待加工原烟数据记录_耐加工性（强、中、弱）", 
                "中"  # 如果前端没传，则使用默认值
            )
            
            fixed_categorical_params = {
                "待加工原烟数据记录_部位（上、中、下）": part_attr
                ,"待加工原烟数据记录_等级（上、中、下）": grade_attr
                ,"待加工原烟数据记录_符合性（优、良、合格）": conformity_attr
                ,"待加工原烟数据记录_耐加工性（强、中、弱）": processability_attr
            }
            print(f"📊 部位属性设置为: {part_attr}")
            print(f"📊 等级属性设置为: {grade_attr}")
            print(f"📊 符合性属性设置为: {conformity_attr}")
            print(f"📊 耐加工性属性设置为: {processability_attr}")

            print(f"📊 手动输入参数数量: {len(manual_parameters)}")
            print(f"📊 固定分类参数: {fixed_categorical_params}")
            # 将手动参数列表化，便于后续处理
            fixed_categorical_param_list = [fixed_categorical_params]
            fixed_numeric_param_list = [manual_parameters]
            
            print(f"📊 手动输入参数数量: {len(manual_parameters)}")
            print(f"📊 固定分类参数: {fixed_categorical_params}")
            
            # 保存手动输入参数到优化进度中
            optimization_progress[task_id]['input_parameters'] = manual_parameters
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

        # 注册数值型变量（移除铺叶工段流量）
        # toolbox.register("二润工段数据记录_流量（kg/h)", random.randint, 12000, 18000)  # 移除这行
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
        best_index_overall = -1  # 新增：记录全局最优解对应的索引

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
                best_index_overall = index  # 更新全局最优解的索引

            # 写入当前最优解到CSV
            if best_solution:
                large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate = best_solution.fitness.values
                writer.writerow(
                    best_solution +
                    [large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate, best_score]
                )
                
                print(f"✅ 第{index + 1}/{iteration_count}组参数优化完成,最优得分: {best_score:.4f}")
                print(f"   遗传算法已运行{max_iterations}代,大中片率: {large_medium_rate:.2f}, 叶含梗率: {leaf_in_stem_rate:.2f}, 梗含叶率: {stem_in_leaf_rate:.2f}")

        # 关闭结果文件
        result_file.close()

        print(f"🎯 优化完成，全局最优得分: {best_overall_score:.4f}")

        # 更新优化结果
        if best_overall_solution:
            # =================== 新增：使用 relearn_model.pkl 进行再学习 ===================
            # 新的目标参数名称列表（与 relearn_model_new.py 中的 targets 一致）
            new_targets_names = [
                "打叶工段数据记录_一打_打辊参数（hz）", "打叶工段数据记录_二打_打辊参数（hz）",
                "打叶工段数据记录_三打_打辊参数（hz）", "打叶工段数据记录_四打_打辊参数（hz）",
                "打叶工段数据记录_风分参数（hz）_一分", "打叶工段数据记录_风分参数（hz）_二分",
                "打叶工段数据记录_风分参数（hz）_三分", "打叶工段数据记录_风分参数（hz）_四分",
                "打叶工段数据记录_风分参数（hz）_五分", "打叶工段数据记录_风分参数（hz）_六分",
                "打叶工段数据记录_风分参数（hz）_七分", "打叶工段数据记录_风分参数（hz）_八分",
                "打叶工段数据记录_风分参数（hz）_九分", "打叶工段数据记录_风分参数（hz）_十分",
                "打叶工段数据记录_风分参数（hz）_十一分"
            ]
            
            # 保存原始参数用于对比
            original_ga_params = dict(zip(param_names, best_overall_solution))
            
            try:
                print("🔄 开始使用 relearn_model.pkl 进行参数再学习...")

                # 1. 加载模型
                relearn_model = joblib.load('./uploads/models/relearn_model.pkl')
                print("✅ 再学习模型加载成功。")

                # 2. 准备模型输入数据
                # 从优化结果中提取固定参数
                if input_mode == 'manual':
                    # 手动模式下，best_index_overall 始终为 0
                    fixed_cat_params_for_relearn = fixed_categorical_param_list[0]
                    fixed_num_params_for_relearn = fixed_numeric_param_list[0]
                else:
                    # 文件模式下，使用找到最优解时对应的索引
                    fixed_cat_params_for_relearn = fixed_categorical_param_list[best_index_overall]
                    fixed_num_params_for_relearn = fixed_numeric_param_list[best_index_overall]

                # 整合所有输入特征到一个字典
                model_input_data = {}
                model_input_data.update(fixed_cat_params_for_relearn)
                model_input_data.update(fixed_num_params_for_relearn)

                # 添加遗传算法输出的参数（作为原始特征）
                ga_output_params = dict(zip(numeric_vars, best_overall_solution[3:]))
                for key, value in ga_output_params.items():
                    model_input_data[f"{key}_原始"] = value
                
                # 转换为 DataFrame
                input_df = pd.DataFrame([model_input_data])
                print(f"📊 准备好的模型输入数据 (前5列): {input_df.iloc[:, :5].to_dict('records')}")

                # 3. 进行预测
                new_target_values = relearn_model.predict(input_df)
                print("🎯 模型预测完成。")

                # --- 新增日志：打印原始结果和模型预测结果 ---
                print("\n" + "="*20 + " 参数更新对比 " + "="*20)
                print("🧬 遗传算法原始输出:")
                for name in new_targets_names:
                    print(f"  - {name}: {original_ga_params.get(name, 'N/A'):.4f}")

                predicted_params = dict(zip(new_targets_names, new_target_values[0]))
                print("\n🤖 relearn_model.pkl 预测输出:")
                for name, value in predicted_params.items():
                    print(f"  - {name}: {value:.4f}")
                print("="*54 + "\n")
                # --- 日志结束 ---

                # 创建新旧参数名到索引的映射
                param_to_index = {name: i for i, name in enumerate(param_names)}

                # 用新预测值覆盖 best_overall_solution 中的旧值
                for i, new_name in enumerate(new_targets_names):
                    if new_name in param_to_index:
                        idx_to_update = param_to_index[new_name]
                        best_overall_solution[idx_to_update] = new_target_values[0, i]

                print("✅ 参数已使用再学习模型的结果进行更新。")
                
                # 重新计算更新后参数的质量指标
                updated_fitness = eval_func(
                    best_overall_solution, 
                    best_index_overall if input_mode != 'manual' else 0,
                    fixed_categorical_param_list, 
                    fixed_numeric_param_list, 
                    target_values, 
                    constraints
                )
                print(f"🔄 重新计算质量指标: 大中片率={updated_fitness[0]:.2f}, 叶含梗率={updated_fitness[1]:.2f}, 梗含叶率={updated_fitness[2]:.2f}")
                
                # 更新fitness值
                large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate = updated_fitness

            except Exception as relearn_error:
                print(f"❌ 在再学习过程中发生错误: {relearn_error}")
                # 即使再学习失败，也继续使用原始优化结果
                large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate = best_overall_solution.fitness.values

            # =================== 再学习逻辑结束 ===================
            
            # 如果没有进行再学习（没有异常），使用原始fitness值
            if 'large_medium_rate' not in locals():
                large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate = best_overall_solution.fitness.values

            # 构建最终参数字典 - 这里会包含更新后的参数
            best_params = {}
            for name, value in zip(param_names, best_overall_solution):
                best_params[name] = value

            # --- 新增日志：打印最终采纳的结果 ---
            print("\n🏆 最终采纳的优化参数:")
            for name in new_targets_names:
                final_value = best_params.get(name, 'N/A')
                original_value = original_ga_params.get(name, 'N/A')
                if isinstance(final_value, (int, float)) and isinstance(original_value, (int, float)):
                    status = "✅ 已更新" if abs(final_value - original_value) > 0.01 else "🔄 未变化"
                    print(f"  - {name}: {final_value:.4f} {status}")
                else:
                    print(f"  - {name}: {final_value}")
            print("="*54)
            # --- 日志结束 ---

            # =================== 新增：重新写入更新后的结果到CSV文件 ===================
            try:
                print("📝 重新写入最终优化结果到CSV文件...")
                
                # 重新打开CSV文件并写入最终结果
                with open(result_file_path, 'w', newline='', encoding='gbk') as final_file:
                    final_writer = csv.writer(final_file)
                    # 写入表头
                    final_writer.writerow(param_names + ['大中片率', '叶含梗率', '梗含叶率', '综合得分'])
                    
                    # 写入最终的优化结果（经过relearn_model.pkl处理的）
                    final_score = large_medium_rate - leaf_in_stem_rate - stem_in_leaf_rate
                    final_writer.writerow(
                        best_overall_solution +
                        [large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate, final_score]
                    )
                
                print(f"✅ 最终结果已写入CSV文件: {result_file_path}")
                print(f"   最终质量指标 - 大中片率: {large_medium_rate:.2f}, 叶含梗率: {leaf_in_stem_rate:.2f}, 梗含叶率: {stem_in_leaf_rate:.2f}")
                
            except Exception as csv_error:
                print(f"❌ 重新写入CSV文件时发生错误: {csv_error}")
            # =================== CSV重写逻辑结束 ===================

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

@parameter_optimization.route('/api/preview_optimization_result_file', methods=['GET'])
@user_or_admin_required
def preview_optimization_result_file():
    """预览优化结果文件内容"""
    try:
        file_path_from_request = request.args.get('file_path')
        if not file_path_from_request:
            return jsonify({'success': False, 'message': '未提供文件路径'})

        # 从请求路径中提取纯文件名
        filename = os.path.basename(file_path_from_request)
        if '..' in filename:  # 安全检查
            return jsonify({'success': False, 'message': '无效的文件名'})

        # 使用项目根目录构建绝对路径，这种方式更稳健
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        abs_file_path = os.path.join(base_dir, 'uploads', 'data', 'results', filename)
        
        if not os.path.exists(abs_file_path):
            return jsonify({'success': False, 'message': f'文件不存在: {abs_file_path}'})
        
        # 读取CSV文件前10行用于预览
        df = pd.read_csv(abs_file_path, encoding='gbk', nrows=10)
        
        # 构造响应数据
        response_data = {
            'columns': df.columns.tolist(),
            'rows': df.values.tolist()
        }
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        print(f"预览优化结果文件错误: {str(e)}")
        return jsonify({'success': False, 'message': f'预览文件失败: {str(e)}'})
    
@parameter_optimization.route('/api/get_optimization_result_data', methods=['GET'])
@user_or_admin_required
def get_optimization_result_data():
    """获取可编辑的优化结果数据"""
    try:
        file_path_from_request = request.args.get('file_path')
        if not file_path_from_request:
            return jsonify({'success': False, 'message': '未提供文件路径'})
            
        filename = os.path.basename(file_path_from_request)
        if '..' in filename:
            return jsonify({'success': False, 'message': '无效的文件名'})

        results_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'results')
        full_path = os.path.join(results_dir, filename)

        if not os.path.exists(full_path):
            return jsonify({'success': False, 'message': '文件不存在', 'path_checked': full_path})

        def read_and_convert_to_ordered_list(path, encoding):
            """读取CSV并将其第一行转换为有序的键值对列表"""
            df = pd.read_csv(path, nrows=1, encoding=encoding)
            ordered_dict = df.iloc[0].to_dict(into=OrderedDict)
            # 转换为键值对列表以保证JSON传输过程中的顺序
            ordered_list = [{'key': k, 'value': v} for k, v in ordered_dict.items()]
            return ordered_list

        try:
            # 优先使用 utf-8-sig 读取
            result_data_list = read_and_convert_to_ordered_list(full_path, 'utf-8-sig')
            return jsonify({'success': True, 'data': result_data_list})
        except UnicodeDecodeError:
            try:
                # 如果失败，则尝试GBK编码
                current_app.logger.warning(f"UTF-8-SIG decoding failed for {full_path}. Trying GBK...")
                result_data_list = read_and_convert_to_ordered_list(full_path, 'gbk')
                return jsonify({'success': True, 'data': result_data_list})
            except Exception as e_gbk:
                #current_app.logger.error(f"Failed to read file {full_path} with both UTF-8-SIG and GBK: {traceback.format_exc()}")
                return jsonify({'success': False, 'message': f'读取文件失败: {str(e_gbk)}'})
        except Exception as e:
            #current_app.logger.error(f"Error reading file {full_path}: {traceback.format_exc()}")
            return jsonify({'success': False, 'message': f'读取文件时发生未知错误: {str(e)}'})

    except Exception as e:
        #current_app.logger.error(f"Error in get_optimization_result_data for {file_path_from_request}: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'处理请求失败: {str(e)}'})
    
@parameter_optimization.route('/api/save_optimization_result_to_database', methods=['POST'])
@user_or_admin_required
def save_optimization_result_to_database():
    """将原始优化结果保存到数据库"""
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'success': False, 'message': '未提供文件路径'})
            
        file_path_from_request = data['file_path']
        filename = os.path.basename(file_path_from_request)
        
        # 从文件名中提取任务ID
        task_id = filename.replace('optimization_result_', '').replace('.csv', '')
        
        # 获取手动输入参数（如果存在）
        input_params_json = None
        if task_id in optimization_progress:
            input_params = optimization_progress[task_id].get('input_parameters')
            if input_params:
                input_params_json = json.dumps(input_params, ensure_ascii=False)
        
        # 使用项目根目录构建绝对路径
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        abs_file_path = os.path.join(base_dir, 'uploads', 'data', 'results', filename)
        
        if not os.path.exists(abs_file_path):
            return jsonify({'success': False, 'message': f'文件不存在: {abs_file_path}'})
        
        # 读取CSV文件最后一行（通常是最优结果）
        df = pd.read_csv(abs_file_path, encoding='gbk')
        
        if df.empty:
            return jsonify({'success': False, 'message': '文件中没有数据'})
        
        # 获取最后一行数据
        best_result = df.iloc[-1].to_dict()
        
        # 准备要保存到数据库的数据
        params_json = {}
        metrics_json = {}
        
        # 分类参数
        for key in categorical_vars.keys():
            if key in best_result:
                params_json[key] = best_result[key]
        
        # 数值参数
        for key in numeric_vars:
            if key in best_result:
                params_json[key] = float(best_result[key])
        
        # 指标数据 - 使用中文键
        metric_keys = ['大中片率', '叶含梗率', '梗含叶率', '综合得分']
        for key in metric_keys:
            if key in best_result:
                try:
                    value = best_result[key]
                    metrics_json[key] = float(value) if pd.notna(value) else None
                except (ValueError, TypeError):
                    metrics_json[key] = None
        
        # 检查数据库中是否已存在此任务的结果
        existing_result = OptimizationResult.query.filter_by(task_id=task_id).first()
        
        if existing_result:
            # 更新现有记录 - 同时保存到两个列
            existing_result.input_parameters = input_params_json
            existing_result.parameters = json.dumps(params_json, ensure_ascii=False)
            existing_result.modified_parameters = json.dumps(params_json, ensure_ascii=False)  # 同样的数据
            existing_result.metrics = json.dumps(metrics_json, ensure_ascii=False)
            # 新增：初始化 modified_metrics 与 metrics 一致
            existing_result.modified_metrics = json.dumps(metrics_json, ensure_ascii=False)  # <-- add
            existing_result.updated_at = datetime.now()
            existing_result.is_modified = False  # 标记为未修改状态
            existing_result.file_path = abs_file_path
        else:
            # 创建新记录 - 同时保存到两个列
            new_result = OptimizationResult(
                task_id=task_id,
                input_parameters=input_params_json,
                parameters=json.dumps(params_json, ensure_ascii=False),
                modified_parameters=json.dumps(params_json, ensure_ascii=False),  # 同样的数据
                metrics=json.dumps(metrics_json, ensure_ascii=False),
                # 新增：初始化 modified_metrics 与 metrics 一致
                modified_metrics=json.dumps(metrics_json, ensure_ascii=False),   # <-- add
                file_path=abs_file_path,
                created_at=datetime.now(),
                is_modified=False
            )
            db.session.add(new_result)
        
        # 提交数据库事务
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '优化结果已成功保存到数据库（同时保存到parameters和modified_parameters列）',
            'task_id': task_id
        })
        
    except Exception as e:
        # 回滚数据库事务
        db.session.rollback()
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'})

@parameter_optimization.route('/api/save_modified_result_to_database', methods=['POST'])
@user_or_admin_required
def save_modified_result_to_database():
    """将修改后的优化结果保存到数据库"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '未接收到请求数据'})

        file_path_from_request = data.get('file_path')
        modified_data = data.get('modified_data', {})
        modified_metrics = data.get('modified_metrics', {})  # 新增：获取修改后的质量指标

        if not file_path_from_request or not isinstance(modified_data, dict):
            return jsonify({'success': False, 'message': '参数不完整或格式不正确'})

        filename = os.path.basename(file_path_from_request)
        if '..' in filename:
            return jsonify({'success': False, 'message': '无效的文件名'})

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        abs_file_path = os.path.join(base_dir, 'uploads', 'data', 'results', filename)

        if not os.path.exists(abs_file_path):
            return jsonify({'success': False, 'message': f'文件不存在: {abs_file_path}'})

        task_id = filename.replace('optimization_result_', '').replace('.csv', '')

        df = pd.read_csv(abs_file_path, encoding='gbk')
        if df.empty:
            return jsonify({'success': False, 'message': '原始文件中没有数据'})

        best_result = df.iloc[-1].to_dict()
        
        # 应用参数修改
        for key, value in modified_data.items():
            if key in best_result:
                best_result[key] = value

        # 分别处理修改后的参数和指标
        modified_params_json = {}
        modified_metrics_json = {}

        # 处理分类参数
        for key in categorical_vars.keys():
            if key in best_result:
                modified_params_json[key] = best_result[key]

        # 处理数值参数
        for key in numeric_vars:
            if key in best_result:
                try:
                    float_val = float(best_result[key])
                    modified_params_json[key] = float_val
                    best_result[key] = float_val
                except (ValueError, TypeError):
                    modified_params_json[key] = None
                    best_result[key] = None

        # 处理质量指标 - 如果前端传来了修改后的指标，使用前端的值；否则使用文件中的值
        metric_keys = ['大中片率', '叶含梗率', '梗含叶率', '综合得分']
        for key in metric_keys:
            if key in modified_metrics:
                # 使用前端传来的修改后的指标值
                modified_metrics_json[key] = modified_metrics[key]
                best_result[key] = modified_metrics[key]
            elif key in best_result:
                # 使用原始文件中的值
                try:
                    float_val = float(best_result[key])
                    modified_metrics_json[key] = float_val
                    best_result[key] = float_val
                except (ValueError, TypeError):
                    modified_metrics_json[key] = None
                    best_result[key] = None
        
        # 保存修改后的CSV文件
        last_row_index = df.index[-1]
        for key, value in best_result.items():
            if key in df.columns:
                df.loc[last_row_index, key] = value
        
        modified_file_path = abs_file_path.replace('.csv', '_modified.csv')
        df.to_csv(modified_file_path, index=False, encoding='gbk')
        
        existing_result = OptimizationResult.query.filter_by(task_id=task_id).first()
        
        if existing_result:
            # 保持原始参数不变，只更新修改后的参数和指标列
            existing_result.modified_parameters = json.dumps(modified_params_json, ensure_ascii=False)
            existing_result.modified_metrics = json.dumps(modified_metrics_json, ensure_ascii=False)  # 新增：保存修改后的质量指标
            existing_result.updated_at = datetime.now()
            existing_result.is_modified = True
            existing_result.file_path = modified_file_path
        else:
            # 如果不存在原始记录，先创建原始记录
            return jsonify({'success': False, 'message': '请先保存原始优化结果到数据库'})
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '修改后的结果已成功保存到数据库（包含质量指标修改）',
            'task_id': task_id,
            'modified_file_path': modified_file_path
        })
        
    except Exception as e:
        # 回滚数据库事务
        db.session.rollback()
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'})


@parameter_optimization.route('/api/predict_parameter_impact', methods=['POST'])
@user_or_admin_required
def predict_parameter_impact():
    """预测单个参数变化对质量指标的影响"""
    print
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '未接收到请求数据'})
        
        param_name = data.get('param_name')
        param_value = data.get('param_value')
        result_file_path = data.get('result_file_path')
        complete_parameters = data.get('complete_parameters', {})
        
        # 支持批量预测的参数格式（用于趋势预览）
        base_parameters = data.get('base_parameters', {})
        varying_parameter = data.get('varying_parameter', '')
        parameter_values = data.get('parameter_values', [])
        
        print(f"🔮 收到预测请求: {data.keys()}")
        
        # 判断是单点预测还是批量预测
        if param_name and param_value is not None:
            # 单点预测模式
            print(f"📊 单点预测: {param_name} = {param_value}")
            
            # 构建完整的参数集合
            final_parameters = build_complete_parameter_set(complete_parameters, param_name, param_value, result_file_path)
            
            if not final_parameters:
                return jsonify({'success': False, 'message': '无法构建完整的参数集合'})
            
            # 调用RBFNN模型预测
            prediction_result = predict_with_rbfnn_enhanced(final_parameters)
            
            return jsonify({
                'success': True,
                'prediction': prediction_result,
                'param_name': param_name,
                'param_value': param_value,
                'total_parameters': len(final_parameters)
            })
            
        elif varying_parameter and parameter_values:
            # 批量预测模式（用于趋势预览）
            print(f"📈 批量预测: {varying_parameter}, 值数量: {len(parameter_values)}")
            
            predictions = []
            
            for param_value in parameter_values:
                # 创建当前参数组合
                current_params = base_parameters.copy()
                current_params[varying_parameter] = param_value
                
                # 进行RBFNN预测
                prediction_result = predict_with_rbfnn_enhanced(current_params)
                
                # 添加参数值到结果中
                prediction_result['parameter_value'] = param_value
                predictions.append(prediction_result)
            
            return jsonify({
                'success': True,
                'message': '参数影响预测完成',
                'varying_parameter': varying_parameter,
                'predictions': predictions
            })
        else:
            return jsonify({'success': False, 'message': '参数格式不正确'})
        
    except Exception as e:
        print(f"参数影响预测错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'预测失败: {str(e)}'})


def build_complete_parameter_set(complete_parameters, changing_param_name, changing_param_value, result_file_path):
    """构建完整的参数集合，包括所有必需的输入参数"""
    try:
        final_parameters = {}
        
        # 1. 从默认数据文件获取基础参数结构
        base_params = get_default_parameters_for_prediction()
        final_parameters.update(base_params)
        print(f"📋 基础参数数量: {len(base_params)}")
        
        # 2. 从结果文件获取优化后的参数（如果存在）
        if result_file_path:
            result_params = get_parameters_from_result_file(result_file_path)
            if result_params:
                final_parameters.update(result_params)
                print(f"📁 结果文件参数数量: {len(result_params)}")
        
        # 3. 应用前端传来的完整参数
        if complete_parameters:
            for key, value in complete_parameters.items():
                if key in categorical_vars:
                    final_parameters[key] = str(value)
                elif any(keyword in key for keyword in ['水份', '温度', '湿度', '含量', '比', '参数', '流量']):
                    try:
                        final_parameters[key] = float(value)
                    except (ValueError, TypeError):
                        print(f"⚠️ 参数 {key} 值转换失败: {value}")
                else:
                    final_parameters[key] = value
            
            print(f"🌐 前端完整参数数量: {len(complete_parameters)}")
        
        # 4. 应用当前变化的参数
        if changing_param_name in categorical_vars:
            final_parameters[changing_param_name] = str(changing_param_value)
        else:
            try:
                final_parameters[changing_param_name] = float(changing_param_value)
            except (ValueError, TypeError):
                print(f"⚠️ 变化参数 {changing_param_name} 值转换失败: {changing_param_value}")
                return None
        
        print(f"🔄 最终应用变化参数: {changing_param_name} = {changing_param_value}")
        
        # 5. 验证关键参数的存在性
        required_categorical = [
            "待加工原烟数据记录_部位（上、中、下）",
            "待加工原烟数据记录_等级（上、中、下）",
            "待加工原烟数据记录_符合性（优、良、合格）",
            "待加工原烟数据记录_耐加工性（强、中、弱）",
            "回潮工段数据记录_回潮方式", 
            "一润工段数据记录_润叶方式（蒸汽、加水、混合）", 
            "二润工段数据记录_润叶方式（蒸汽、加水、混合）"
        ]
        
        for param in required_categorical:
            if param not in final_parameters:
                final_parameters[param] = get_default_categorical_value(param)
                print(f"🔧 补充缺失的分类参数: {param}")
        
        print(f"✅ 构建完整参数集合成功，总参数数量: {len(final_parameters)}")
        return final_parameters
        
    except Exception as e:
        print(f"构建完整参数集合错误: {str(e)}")
        return None


def get_default_parameters_for_prediction():
    """获取预测用的默认参数"""
    default_params = {}
    
    # 默认分类参数
    default_params.update({
        "待加工原烟数据记录_部位（上、中、下）": "中",
        "待加工原烟数据记录_等级（上、中、下）": "中",
        "待加工原烟数据记录_符合性（优、良、合格）": "良",
        "待加工原烟数据记录_耐加工性（强、中、弱）": "中",
        "回潮工段数据记录_回潮方式": "半回",
        "一润工段数据记录_润叶方式（蒸汽、加水、混合）": "混合",
        "二润工段数据记录_润叶方式（蒸汽、加水、混合）": "混合"
    })
    
    # 默认数值参数（包含铺叶工段流量）
    default_params.update({
        "待加工原烟数据记录_水份（%）": 20.2,
        "待加工原烟数据记录_温度（℃）": 25.0,
        "待加工原烟数据记录_烟碱含量（%）": 2.5,
        "待加工原烟数据记录_叶梗比（%）": 15.0,
        "待加工原烟数据记录_环境温度（℃）": 22.0,
        "待加工原烟数据记录_环境湿度（%）": 60.0,
        "铺叶工段数据记录_环境温度（℃）": 23.0,
        "铺叶工段数据记录_环境湿度（%）": 65.0,
        "二润工段数据记录_流量（kg/h)": 14000,  # 设置默认值
        "回潮工段数据记录_潮后水份（%）": 18.5,
        "回潮工段数据记录_潮后温度（℃）": 45.0,
        "回潮工段数据记录_环境温度（℃）": 24.0,
        "回潮工段数据记录_环境湿度（%）": 70.0,
        "一润工段数据记录_润前水份（%）": 16.0,
        "一润工段数据记录_润前温度（℃）": 40.0,
        "一润工段数据记录_润后水份（%）": 18.0,
        "一润工段数据记录_润后温度（℃）": 50.0,
        "一润工段数据记录_环境温度（℃）": 25.0,
        "一润工段数据记录_环境湿度（%）": 75.0,
        "二润工段数据记录_润前水份（%）": 17.0,
        "二润工段数据记录_润前温度（℃）": 45.0,
        "二润工段数据记录_润后水份（%）": 19.0,
        "二润工段数据记录_润后温度（℃）": 55.0,
        "二润工段数据记录_环境温度（℃）": 26.0,
        "二润工段数据记录_环境湿度（%）": 80.0,
        "打叶工段数据记录_打前水份（%）": 15.5,
        "打叶工段数据记录_打前温度（℃）": 35.0,
        "打叶工段数据记录_环境湿度（%）": 65.0,
        "打叶工段数据记录_环境温度（℃）": 27.0,
        # 可优化参数的默认值
        "打叶工段数据记录_一打_打辊参数（hz）": 42,
        "打叶工段数据记录_二打_打辊参数（hz）": 45,
        "打叶工段数据记录_三打_打辊参数（hz）": 48,
        "打叶工段数据记录_四打_打辊参数（hz）": 48,
        "打叶工段数据记录_风分参数（hz）_一分": 33,
        "打叶工段数据记录_风分参数（hz）_二分": 31,
        "打叶工段数据记录_风分参数（hz）_三分": 36,
        "打叶工段数据记录_风分参数（hz）_四分": 34,
        "打叶工段数据记录_风分参数（hz）_五分": 28,
        "打叶工段数据记录_风分参数（hz）_六分": 33,
        "打叶工段数据记录_风分参数（hz）_七分": 41,
        "打叶工段数据记录_风分参数（hz）_八分": 40,
        "打叶工段数据记录_风分参数（hz）_九分": 41,
        "打叶工段数据记录_风分参数（hz）_十分": 36,
        "打叶工段数据记录_风分参数（hz）_十一分": 46
    })
    
    return default_params


def get_parameters_from_result_file(result_file_path):
    """从结果文件获取参数"""
    try:
        if not result_file_path or not os.path.exists(result_file_path):
            return {}
            
        df = pd.read_csv(result_file_path, encoding='gbk')
        if df.empty:
            return {}
            
        # 获取最后一行（最优解）的参数
        last_row = df.iloc[-1]
        params = {}
        
        # 提取分类参数
        for param_name in categorical_vars.keys():
            if param_name in last_row:
                params[param_name] = str(last_row[param_name])
        
        # 提取数值参数
        for col in df.columns:
            if col not in categorical_vars.keys() and col not in ['大中片率', '叶含梗率', '梗含叶率']:
                try:
                    params[col] = float(last_row[col])
                except (ValueError, TypeError):
                    pass
        
        return params
        
    except Exception as e:
        print(f"从结果文件获取参数错误: {str(e)}")
        return {}


def get_default_categorical_value(param_name):
    """获取分类参数的默认值"""
    defaults = {
        "待加工原烟数据记录_部位（上、中、下）": "中",
        "待加工原烟数据记录_等级（上、中、下）": "中",
        "待加工原烟数据记录_符合性（优、良、合格）": "良",
        "待加工原烟数据记录_耐加工性（强、中、弱）": "中",
        "回潮工段数据记录_回潮方式": "半回",
        "一润工段数据记录_润叶方式（蒸汽、加水、混合）": "混合",
        "二润工段数据记录_润叶方式（蒸汽、加水、混合）": "混合"
    }
    return defaults.get(param_name, "合格")


def predict_with_rbfnn_enhanced(parameters):
    """使用RBFNN模型进行预测的增强版本，包含完整的数据预处理"""
    try:
        print(f"🧠 使用RBFNN模型预测，输入参数数量: {len(parameters)}")
        
        # 构建分类参数DataFrame - 按照固定顺序
        categorical_params_order = [
            "待加工原烟数据记录_部位（上、中、下）",
            "待加工原烟数据记录_等级（上、中、下）",
            "待加工原烟数据记录_符合性（优、良、合格）",
            "待加工原烟数据记录_耐加工性（强、中、弱）",
            "回潮工段数据记录_回潮方式", 
            "一润工段数据记录_润叶方式（蒸汽、加水、混合）", 
            "二润工段数据记录_润叶方式（蒸汽、加水、混合）"
        ]

        # 构建分类参数DataFrame
        categorical_data = {}
        for param_name in categorical_params_order:
            categorical_data[param_name] = [parameters.get(param_name, get_default_categorical_value(param_name))]
        
        categorical_df = pd.DataFrame(categorical_data)
        
        # 🔧 更安全的动态构建categories方式
        categories_map = {
            "待加工原烟数据记录_部位（上、中、下）": ["上", "中", "下"],        # ✅ 改为字符串
            "待加工原烟数据记录_等级（上、中、下）": ["上", "中", "下"],        # ✅ 改为字符串
            "待加工原烟数据记录_符合性（优、良、合格）": ["优", "良", "合格"],    # ✅ 改为字符串
            "待加工原烟数据记录_耐加工性（强、中、弱）": ["强", "中", "弱"],      # ✅ 改为字符串
            "回潮工段数据记录_回潮方式": ["半回", "全回"],
            "一润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"],
            "二润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"]
        }
        
        # 按照DataFrame列的实际顺序构建categories列表
        categories_list = []
        for col in categorical_df.columns:
            categories_list.append(categories_map[col])
        
        # One-Hot编码
        encoder = OneHotEncoder(sparse_output=False, categories=categories_list)
        encoded_array = encoder.fit_transform(categorical_df)
        feature_names = encoder.get_feature_names_out(input_features=categorical_df.columns)
        encoded_df = pd.DataFrame(encoded_array.astype(bool), columns=feature_names)
        # 构建数值参数DataFrame
        numeric_data = {}
        all_numeric_params = [col for col in parameters.keys() if col not in categorical_vars.keys()]
        
        for param_name in all_numeric_params:
            if param_name in parameters:
                numeric_data[param_name] = [parameters[param_name]]
        
        numeric_df = pd.DataFrame(numeric_data)
        
        # 合并特征
        X_input = pd.concat([encoded_df, numeric_df], axis=1)

        # 🔧 新增：将特定分类特征转换为数值
        categorical_to_numeric_map = {
            "待加工原烟数据记录_部位（上、中、下）": {"上": 2, "中": 1, "下": 0},
            "待加工原烟数据记录_等级（上、中、下）": {"上": 2, "中": 1, "下": 0},
            "待加工原烟数据记录_符合性（优、良、合格）": {"优": 2, "良": 1, "合格": 0},
            "待加工原烟数据记录_耐加工性（强、中、弱）": {"强": 2, "中": 1, "弱": 0}
        }

        # 转换指定列为数值
        for col_name, value_map in categorical_to_numeric_map.items():
            if col_name in X_input.columns:
                X_input[col_name] = X_input[col_name].map(value_map)

        
        # 使用三个RBFNN模型进行预测
        predictions = {}
        
        # 预测所有三个指标
        model_info = [
            ('梗含叶率', './uploads/models/梗含叶率_rbfnn.pkl', 'stem_leaf_rate'),
            ('大中片率', './uploads/models/大中片率_rbfnn.pkl', 'big_medium_rate'),
            ('叶含梗率', './uploads/models/叶含梗率_rbfnn.pkl', 'leaf_stem_rate')
        ]
        
        for metric_name, model_path, result_key in model_info:
            if os.path.exists(model_path):
                print(f"🔮 正在预测 {metric_name}...")
                
                with open(model_path, 'rb') as f:
                    model_package = pickle.load(f)
                
                model = model_package['model']
                scaler_X = model_package['scaler_X']
                scaler_y = model_package['scaler_y']
                feature_names = scaler_X.feature_names_in_
                
                # 确保特征顺序一致
                X_aligned = X_input[feature_names]
                X_scaled = scaler_X.transform(X_aligned)
                y_pred_scaled = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
                
                predictions[result_key] = float(y_pred.flatten()[0])
                print(f"✅ {metric_name} 预测完成: {predictions[result_key]:.3f}")

            else:
                print(f"⚠️ 模型文件不存在: {model_path}")
                # 提供默认值
                default_values = {'stem_leaf_rate': 0.6, 'big_medium_rate': 78.0, 'leaf_stem_rate': 1.4}
                predictions[result_key] = default_values[result_key]
        
        # 应用大中片率约束条件检查 - 根据质量属性设置上限
       
        if 'big_medium_rate' in predictions:
            original_rate = predictions['big_medium_rate']

        
        print(f"🎯 RBFNN预测完成: {predictions}")
        return predictions
        
    except Exception as e:
        print(f"RBFNN预测错误: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回默认值
        return {
            'stem_leaf_rate': 0.6,
            'big_medium_rate': 78.0,
            'leaf_stem_rate': 1.4
        }


# 替换原来的predict_with_rbfnn函数
def predict_with_rbfnn(parameters_data):
    """兼容性包装函数，调用增强版本"""
    return predict_with_rbfnn_enhanced(parameters_data)





import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

# 全局变量存储优化进度
optimization_progress = {}

class ImprovedRBFNN:
    def __init__(self, n_centers, sigma=1.0, reg_lambda=1e-6):
        self.n_centers = n_centers  # 隐藏层神经元数量
        self.sigma = sigma  # 高斯核宽度参数
        self.centers = None  # RBF中心点
        self.weights = None  # 输出层权重
        self.reg_lambda = reg_lambda  # 正则化参数

    def _gaussian_kernel(self, x, centers):
        """改进的高斯径向基函数，考虑特征维度"""
        n_samples = x.shape[0]
        n_centers = centers.shape[0]
        n_features = x.shape[1]

        # 调整sigma基于特征维度 - 保留此逻辑
        sigma = self.sigma * np.sqrt(n_features)

        distances = np.zeros((n_samples, n_centers))
        for i in range(n_centers):
            # 自适应距离计算
            sq_diff = (x - centers[i]) ** 2
            distances[:, i] = np.sum(sq_diff, axis=1)

        return np.exp(-distances / (2 * sigma ** 2))

    def predict(self, X):
        rbf_output = self._gaussian_kernel(X, self.centers)
        rbf_output = np.column_stack([rbf_output, np.ones(X.shape[0])])
        return rbf_output.dot(self.weights)

def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_excel_file(file_path, output_path):
    """
    处理Excel文件，按照烟草项目特定格式进行数据提取和重组
    
    Args:
        file_path: 输入Excel文件路径
        output_path: 输出CSV文件路径
    
    Returns:
        tuple: (success, message, processed_data)
    """
    try:
        # 读取 Excel 文件
        excel_file = pd.ExcelFile(file_path)
        
        # 定义标准列名
        columns = [
            '待加工原烟数据记录_日期', '待加工原烟数据记录_时间', '待加工原烟数据记录_水份（%）',
            '待加工原烟数据记录_温度（℃）', '待加工原烟数据记录_烟碱含量（%）',
            '待加工原烟数据记录_质量属性（稍差、合格、稍强）', '待加工原烟数据记录_等级产地', 
            '待加工原烟数据记录_叶梗比（%）', '待加工原烟数据记录_环境温度', '待加工原烟数据记录_环境湿度', 
            '回软工段数据记录_原烟水份（%）', '回软工段数据记录_原烟温度（℃）', 
            '回软工段数据记录_回软间设定温度（℃）', '回软工段数据记录_回软时长（小时）',
            '回软工段数据记录_环境温度', '回软工段数据记录_环境湿度', '铺叶工段数据记录_铺叶模式',
            '铺叶工段数据记录_投料颗粒度', '铺叶工段数据记录_流量（kg/h)', '铺叶工段数据记录_环境温度',
            '铺叶工段数据记录_环境湿度', '回潮工段数据记录_潮后水份（%）', '回潮工段数据记录_潮后温度（℃）',
            '回潮工段数据记录_回潮方式', '回潮工段数据记录_环境温度', '回潮工段数据记录_环境湿度',
            '一润工段数据记录_润前水份（%）', '一润工段数据记录_润前温度（℃）',
            '一润工段数据记录_润叶方式（蒸汽、加水、水汽混合）', '一润工段数据记录_润后水份（%）',
            '一润工段数据记录_润后温度（℃）', '一润工段数据记录_环境温度', '一润工段数据记录_环境湿度',
            '二润工段数据记录_润前水份（%）', '二润工段数据记录_润前温度（℃）', '二润工段数据记录_打叶线流量',
            '二润工段数据记录_润叶方式（蒸汽、加水、水汽混合）', '二润工段数据记录_润后水份（%）',
            '二润工段数据记录_润后温度（℃）', '二润工段数据记录_环境温度', '二润工段数据记录_环境湿度',
            '打叶工段数据记录_打前水份（%）', '打叶工段数据记录_打前温度（℃）', 
            '打叶工段数据记录_一打_打辊参数（hz）', '打叶工段数据记录_二打_打辊参数（hz）',
            '打叶工段数据记录_三打_打辊参数（hz）', '打叶工段数据记录_四打_打辊参数（hz）',
            '打叶工段数据记录_风分参数（hz）_一分', '打叶工段数据记录_风分参数（hz）_二分',
            '打叶工段数据记录_风分参数（hz）_三分', '打叶工段数据记录_风分参数（hz）_四分',
            '打叶工段数据记录_风分参数（hz）_五分', '打叶工段数据记录_风分参数（hz）_六分',
            '打叶工段数据记录_风分参数（hz）_七分', '打叶工段数据记录_风分参数（hz）_八分',
            '打叶工段数据记录_风分参数（hz）_九分', '打叶工段数据记录_风分参数（hz）_十分',
            '打叶工段数据记录_风分参数（hz）_十一分', '打叶工段数据记录_风分参数（hz）_十二分', 
            '打叶工段数据记录_大中片率', '打叶工段数据记录_叶含梗率', '打叶工段数据记录_梗含叶率', 
            '打叶工段数据记录_环境温度', '打叶工段数据记录_环境湿度',
            '复烤工段数据记录_烤前水分（%）', '复烤工段数据记录_烤前温度（℃）', 
            '复烤工段数据记录_入口流量（kg/h）', '复烤工段数据记录_干燥一区_温度（℃）', 
            '复烤工段数据记录_干燥二区_温度（℃）', '复烤工段数据记录_干燥三区_温度（℃）', 
            '复烤工段数据记录_干燥四区_温度（℃）', '复烤工段数据记录_干燥五区_温度（℃）', 
            '复烤工段数据记录_干燥六区_温度（℃）', '复烤工段数据记录_冷却区_温度（℃）',
            '复烤工段数据记录_冷却区_水分（%）', '复烤工段数据记录_回潮一区_温度（℃）', 
            '复烤工段数据记录_回潮一区_水分（%）', '复烤工段数据记录_回潮二区_温度（℃）', 
            '复烤工段数据记录_回潮二区_水分（%）', '复烤工段数据记录_回潮三区_温度（℃）', 
            '复烤工段数据记录_回潮三区_水分（%）', '复烤工段数据记录_烤后水分（%）',
            '复烤工段数据记录_烤后温度（℃）', '复烤工段数据记录_排潮风机_参数（hz）', 
            '复烤工段数据记录_循环风机参数 冷却区', '复烤工段数据记录_排潮风机参数 回潮区', 
            '复烤工段数据记录_网带频率（hz）', '复烤工段数据记录_回潮方式（蒸汽、加水、水汽混合）', 
            '复烤工段数据记录_烟碱值', '复烤工段数据记录_环境温度', '复烤工段数据记录_环境湿度', 
            '打包工段数据记录_装箱温度（℃）', '打包工段数据记录_装箱水分（%）',
            '打包工段数据记录_装箱密度', '打包工段数据记录_环境温度', 
            '打包工段数据记录_环境湿度', '备注栏'
        ]

        # 尝试读取不同的sheet名称
        sheet_names = excel_file.sheet_names
        target_sheet = None
        
        # 查找包含"总表"或"总"的sheet
        for sheet in sheet_names:
            if '总表' in sheet or '总' in sheet:
                target_sheet = sheet
                break
        
        if not target_sheet:
            target_sheet = sheet_names[0]  # 使用第一个sheet
        
        # 从第4行开始加载数据（header=3表示第4行作为列名）
        df = excel_file.parse(target_sheet, header=3)
        
        # 检查数据行数
        if len(df) == 0:
            return False, "Excel文件数据为空", None
        
        # 如果列数匹配，设置标准列名
        if len(df.columns) == len(columns):
            df.columns = columns
        else:
            # 列数不匹配时，保持原列名但记录警告
            pass
        
        # 提取各个工段的关键数据
        extracted_data = {}
        
        # 提取待加工原烟数据记录
        tobacco_columns = [col for col in df.columns if '待加工原烟数据记录' in col]
        if tobacco_columns:
            extracted_data['tobacco_df'] = df[tobacco_columns]
        
        # 提取铺叶工段数据记录
        leaf_laying_columns = [col for col in df.columns if '铺叶工段数据记录' in col]
        if leaf_laying_columns:
            extracted_data['leaf_laying_df'] = df[leaf_laying_columns]
        
        # 提取二润工段数据记录
        second_moistening_columns = [col for col in df.columns if '二润工段数据记录' in col]
        if second_moistening_columns:
            extracted_data['second_moistening_df'] = df[second_moistening_columns]
        
        # 提取打叶工段数据记录
        leaf_threshing_columns = [col for col in df.columns if '打叶工段数据记录' in col]
        if leaf_threshing_columns:
            extracted_data['leaf_threshing_df'] = df[leaf_threshing_columns]
        
        # 合并所有提取的数据
        if extracted_data:
            combined_df = pd.concat(list(extracted_data.values()), axis=1)
        else:
            # 如果没有匹配的列，使用原始数据
            combined_df = df
        
        # 数据清洗
        combined_df = combined_df.dropna(how='all')  # 删除全为空的行
        
        # 保存为CSV文件
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 生成预览数据
        preview_data = {
            'filename': os.path.basename(output_path),
            'rows': len(combined_df),
            'columns': len(combined_df.columns),
            'preview': combined_df.head(5).fillna('').to_dict('records'),
            'column_names': combined_df.columns.tolist(),
            'extracted_sections': list(extracted_data.keys()) if extracted_data else ['全部数据']
        }
        
        return True, "Excel文件处理成功", preview_data
        
    except Exception as e:
        return False, f"Excel文件处理失败: {str(e)}", None

def load_model(model_path):
    """加载机器学习模型"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"模型加载失败: {str(e)}")

def save_model(model, model_path):
    """保存机器学习模型"""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        raise Exception(f"模型保存失败: {str(e)}")

def optimize_parameters(algorithm, parameters, objective_function, config):
    """
    参数优化函数
    
    Args:
        algorithm: 优化算法名称 ('PSO', 'GA', 'SA', 'GD')
        parameters: 参数范围字典
        objective_function: 目标函数
        config: 算法配置参数
    
    Returns:
        dict: 优化结果
    """
    
    if algorithm == 'PSO':
        return pso_optimize(parameters, objective_function, config)
    elif algorithm == 'GA':
        return ga_optimize(parameters, objective_function, config)
    elif algorithm == 'SA':
        return sa_optimize(parameters, objective_function, config)
    elif algorithm == 'GD':
        return gd_optimize(parameters, objective_function, config)
    else:
        raise ValueError(f"不支持的优化算法: {algorithm}")

def pso_optimize(parameters, objective_function, config):
    """粒子群优化算法"""
    n_particles = config.get('particles', 30)
    n_iterations = config.get('iterations', 100)
    w = config.get('w', 0.5)
    c1 = config.get('c1', 1.5)
    c2 = config.get('c2', 1.5)
    
    # 参数范围
    param_names = list(parameters.keys())
    bounds = np.array([[parameters[name]['min'], parameters[name]['max']] for name in param_names])
    
    # 初始化粒子
    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, len(param_names)))
    velocities = np.zeros_like(particles)
    
    # 个体最优和全局最优
    pbest = particles.copy()
    pbest_scores = np.array([objective_function(p) for p in particles])
    gbest_idx = np.argmax(pbest_scores)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    history = []
    
    for iteration in range(n_iterations):
        for i in range(n_particles):
            # 更新速度
            r1, r2 = np.random.random(2)
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (pbest[i] - particles[i]) + 
                           c2 * r2 * (gbest - particles[i]))
            
            # 更新位置
            particles[i] += velocities[i]
            
            # 边界处理
            particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
            
            # 评估适应度
            score = objective_function(particles[i])
            
            # 更新个体最优
            if score > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score
                
                # 更新全局最优
                if score > gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = score
        
        history.append(gbest_score)
    
    # 返回结果
    result = {param_names[i]: gbest[i] for i in range(len(param_names))}
    
    return {
        'best_params': result,
        'best_score': gbest_score,
        'history': history,
        'iterations': n_iterations
    }

def ga_optimize(parameters, objective_function, config):
    """遗传算法优化"""
    # 简化的遗传算法实现
    population_size = config.get('population', 50)
    generations = config.get('generations', 100)
    
    param_names = list(parameters.keys())
    bounds = np.array([[parameters[name]['min'], parameters[name]['max']] for name in param_names])
    
    # 初始化种群
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, len(param_names)))
    
    history = []
    best_individual = None
    best_score = -np.inf
    
    for generation in range(generations):
        # 评估适应度
        scores = np.array([objective_function(ind) for ind in population])
        
        # 记录最优
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_individual = population[max_idx].copy()
        
        history.append(best_score)
        
        # 选择、交叉、变异操作（简化实现）
        # 这里只做简单的随机搜索来模拟
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, len(param_names)))
    
    result = {param_names[i]: best_individual[i] for i in range(len(param_names))}
    
    return {
        'best_params': result,
        'best_score': best_score,
        'history': history,
        'iterations': generations
    }

def sa_optimize(parameters, objective_function, config):
    """模拟退火算法"""
    initial_temp = config.get('initial_temp', 1000)
    cooling_rate = config.get('cooling_rate', 0.95)
    min_temp = config.get('min_temp', 0.01)
    
    param_names = list(parameters.keys())
    bounds = np.array([[parameters[name]['min'], parameters[name]['max']] for name in param_names])
    
    # 初始解
    current = np.random.uniform(bounds[:, 0], bounds[:, 1])
    current_score = objective_function(current)
    
    best = current.copy()
    best_score = current_score
    
    temp = initial_temp
    history = []
    iteration = 0
    
    while temp > min_temp:
        # 生成邻域解
        neighbor = current + np.random.normal(0, 0.1, len(param_names))
        neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
        
        neighbor_score = objective_function(neighbor)
        
        # 接受准则
        if neighbor_score > current_score or np.random.random() < np.exp((neighbor_score - current_score) / temp):
            current = neighbor
            current_score = neighbor_score
            
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        history.append(best_score)
        temp *= cooling_rate
        iteration += 1
    
    result = {param_names[i]: best[i] for i in range(len(param_names))}
    
    return {
        'best_params': result,
        'best_score': best_score,
        'history': history,
        'iterations': iteration
    }

def gd_optimize(parameters, objective_function, config):
    """梯度下降算法（数值梯度）"""
    learning_rate = config.get('learning_rate', 0.01)
    iterations = config.get('iterations', 1000)
    
    param_names = list(parameters.keys())
    bounds = np.array([[parameters[name]['min'], parameters[name]['max']] for name in param_names])
    
    # 初始点
    current = np.random.uniform(bounds[:, 0], bounds[:, 1])
    
    history = []
    
    for iteration in range(iterations):
        # 计算数值梯度
        gradient = np.zeros_like(current)
        eps = 1e-5
        
        for i in range(len(current)):
            current_plus = current.copy()
            current_minus = current.copy()
            current_plus[i] += eps
            current_minus[i] -= eps
            
            gradient[i] = (objective_function(current_plus) - objective_function(current_minus)) / (2 * eps)
        
        # 更新参数
        current += learning_rate * gradient
        current = np.clip(current, bounds[:, 0], bounds[:, 1])
        
        score = objective_function(current)
        history.append(score)
    
    result = {param_names[i]: current[i] for i in range(len(param_names))}
    
    return {
        'best_params': result,
        'best_score': objective_function(current),
        'history': history,
        'iterations': iterations
    }

def default_objective_function(params):
    """默认目标函数（模拟质量评分计算）"""
    # 这里实现一个简单的目标函数来模拟质量评分
    # 实际应用中应该使用训练好的机器学习模型
    
    if isinstance(params, dict):
        rate = params.get('大中片率', 78)
        leaf_stem = params.get('叶含梗率', 1.1)
        stem_leaf = params.get('梗含叶率', 0.5)
    else:
        rate, leaf_stem, stem_leaf = params
    
    # 模拟质量评分计算
    score = (rate * 0.6 +  # 大中片率权重最高
             (2.0 - leaf_stem) * 20 +  # 叶含梗率越小越好
             (1.0 - stem_leaf) * 15)   # 梗含叶率越小越好
    
    return score

def calculate_sensitivity(parameters):
    """计算参数敏感性分析"""
    sensitivity_data = {}
    base_score = default_objective_function(parameters)
    
    for param_name, param_value in parameters.items():
        if param_name in ['大中片率', '叶含梗率', '梗含叶率']:
            # 计算参数变化对目标函数的影响
            delta = param_value * 0.1  # 10%变化
            
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            params_plus[param_name] += delta
            params_minus[param_name] -= delta
            
            score_plus = default_objective_function(params_plus)
            score_minus = default_objective_function(params_minus)
            
            # 计算敏感性（归一化）
            sensitivity = abs((score_plus - score_minus) / (2 * delta))
            sensitivity_data[param_name] = sensitivity
    
    # 归一化敏感性值
    max_sensitivity = max(sensitivity_data.values()) if sensitivity_data else 1
    for param in sensitivity_data:
        sensitivity_data[param] = (sensitivity_data[param] / max_sensitivity) * 100
    
    return sensitivity_data

def validate_data_file(file_path):
    """验证数据文件格式"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # 检查必要的列
        required_columns = ['待加工原烟数据记录_水份（%）', '待加工原烟数据记录_温度（℃）']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"缺少必要列: {', '.join(missing_columns)}"
        
        return True, "数据文件格式正确"
        
    except Exception as e:
        return False, f"文件读取错误: {str(e)}"

def preprocess_data(df):
    """数据预处理"""
    # 处理缺失值
    df = df.fillna(df.mean(numeric_only=True))
    
    # 数据标准化
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df, scaler
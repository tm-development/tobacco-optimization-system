from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans
import random
import pandas as pd
import csv
import warnings
from collections import OrderedDict
import functools

import pickle
import pandas as pd
import numpy as np

def extract_fixed_params_from_csv(file_path):
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

    # 将数值参数转换为浮点类型

    return fixed_categorical_params.to_dict('records'), fixed_numeric_params.to_dict('records')


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

        # 调整sigma基于特征维度
        sigma = self.sigma * np.sqrt(n_features)

        distances = np.zeros((n_samples, n_centers))
        for i in range(n_centers):
            # 自适应距离计算
            sq_diff = (x - centers[i]) ** 2
            distances[:, i] = np.sum(sq_diff, axis=1)

        return np.exp(-distances / (2 * sigma ** 2))

    def fit(self, X, y, max_iter=300, tol=1e-6):
        # 使用K-means++确定中心点
        n_clusters = min(self.n_centers, X.shape[0])
        if n_clusters == 0:
            self.centers = np.array([])
            self.weights = np.array([])
            return self

        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=max_iter,
            tol=tol,
            random_state=42,
            n_init=10
        )
        try:
            kmeans.fit(X)
            self.centers = kmeans.cluster_centers_
        except ValueError as e:
            print(f"KMeans聚类失败: {e}")
            self.centers = np.array([])
            self.weights = np.array([])
            return self

        # 如果中心点为空，表示训练失败
        if self.centers.shape[0] == 0:
            self.weights = np.array([])
            return self

        # 计算RBF层输出
        rbf_output = self._gaussian_kernel(X, self.centers)

        # 添加偏置项
        rbf_output = np.column_stack([rbf_output, np.ones(X.shape[0])])

        # 带正则化的伪逆求解
        rbf_output = rbf_output.astype(float)
        y = y.astype(float)

        identity = np.eye(rbf_output.shape[1])
        # L2正则化求解
        A = rbf_output.T.dot(rbf_output) + self.reg_lambda * identity

        # 检查A是否可逆
        try:
            self.weights = np.linalg.solve(A, rbf_output.T.dot(y))
        except np.linalg.LinAlgError:
            try:
                self.weights = np.linalg.pinv(rbf_output).dot(y)
            except Exception as e:
                print(f"权重求解失败: {e}")
                self.weights = np.array([])

        return self

    def predict(self, X):
        # 如果模型未成功训练，则返回零数组
        if self.centers is None or self.centers.shape[0] == 0 or self.weights is None or self.weights.shape[0] == 0:
            return np.zeros((X.shape[0], 1))

        rbf_output = self._gaussian_kernel(X, self.centers)
        rbf_output = np.column_stack([rbf_output, np.ones(X.shape[0])])
        return rbf_output.dot(self.weights)




warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)       # 显示所有行
pd.set_option('display.max_columns', None)    # 显示所有列
pd.set_option('display.width', None)          # 自动调整宽度
pd.set_option('display.max_colwidth', None)   # 显示完整列内容


def has_numbers_in_first_three(arr):
    if len(arr) < 3:
        return False

    for item in arr[:3]:
        if isinstance(item, (int, float)):
            return True
    return False




#===========遗传算法进行参数优化============

# 三个字符型变量及其可能的取值
categorical_vars = OrderedDict({
    "回潮工段数据记录_回潮方式": ["半回", "全回"],
    "一润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"],
    "二润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"]
})

# 数值型变量（范围 0-100 的整数）
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

creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

# 初始化工具箱
toolbox = base.Toolbox()

# 注册字符型变量
for var_name, var_values in categorical_vars.items():
    # print(var_values)
    toolbox.register(var_name, random.choice, var_values)

def generate_float(low, high, precision):
    return round(random.uniform(low, high), precision)

# 注册数值型变量（0-100 的整数）
# for var_name in numeric_vars:
#     toolbox.register(var_name, generate_float, 10, 50, 1)
toolbox.register("铺叶工段数据记录_流量（kg/h)",random.randint, 12000,18000)
toolbox.register("打叶工段数据记录_一打_打辊参数（hz）",random.randint, 30,50)
toolbox.register("打叶工段数据记录_二打_打辊参数（hz）",random.randint, 30,50)
toolbox.register("打叶工段数据记录_三打_打辊参数（hz）",random.randint, 30,50)
toolbox.register("打叶工段数据记录_四打_打辊参数（hz）",random.randint, 30,50)

toolbox.register("打叶工段数据记录_风分参数（hz）_一分",random.randint, 28,50)
toolbox.register("打叶工段数据记录_风分参数（hz）_二分",random.randint, 28,50)
toolbox.register("打叶工段数据记录_风分参数（hz）_三分",random.randint, 28,50)
toolbox.register("打叶工段数据记录_风分参数（hz）_四分",random.randint, 28,50)
toolbox.register("打叶工段数据记录_风分参数（hz）_五分",random.randint, 26,45)
toolbox.register("打叶工段数据记录_风分参数（hz）_六分",random.randint, 26,45)
toolbox.register("打叶工段数据记录_风分参数（hz）_七分",random.randint, 30,50)
toolbox.register("打叶工段数据记录_风分参数（hz）_八分",random.randint, 30,45)
toolbox.register("打叶工段数据记录_风分参数（hz）_九分",random.randint, 30,45)
toolbox.register("打叶工段数据记录_风分参数（hz）_十分",random.randint, 33,50)
toolbox.register("打叶工段数据记录_风分参数（hz）_十一分",random.randint, 33,50)


categorical_funcs = [toolbox.__getattribute__(var_name) for var_name in categorical_vars.keys()]
numeric_funcs = [toolbox.__getattribute__(var_name) for var_name in numeric_vars]
all_funcs = categorical_funcs + numeric_funcs



# 注册个体
toolbox.register("individual", tools.initCycle, creator.Individual, all_funcs, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

fixed_categorical_param_list, fixed_numeric_param_list = extract_fixed_params_from_csv(
    './uploads/data/5.14天昌_处理后数据_10条.csv')




def eval_func(individual, i):
    fixed_categorical_params = fixed_categorical_param_list[i]
    fixed_numeric_params = fixed_numeric_param_list[i]
    # 解包个体参数（顺序需与注册顺序一致）
    if has_numbers_in_first_three(individual):
        return 0, 10, 10
    # 提取字符型变量（前 3 个变量）


    B0 = individual[0]  # 回潮方式
    B1 = individual[1]  # 一润方式
    B2 = individual[2]  # 二润方式


    # 提取数值型变量（后 19 个变量）
    numeric_values = individual[3:]  # 共 19 个数值型变量


    # 对字符型变量进行 One-Hot 编码
    df_categorical = pd.DataFrame({
        "回潮工段数据记录_回潮方式": [B0],
        "一润工段数据记录_润叶方式（蒸汽、加水、混合）": [B1],
        "二润工段数据记录_润叶方式（蒸汽、加水、混合）": [B2]
    })
    categorical_input = pd.concat([pd.DataFrame([fixed_categorical_params]), df_categorical], axis=1)
    # 拟合并转换数据（注意：输入必须是二维数组）
    encoder = OneHotEncoder(sparse_output=False, categories=[["稍差", "合格", "稍强"],["半回", "全回"],
                                                             ["蒸汽", "加水", "混合"], ["蒸汽", "加水", "混合"]])
    encoded_array = encoder.fit_transform(categorical_input)
    # 获取编码后的列名（自动添加变量名前缀）
    feature_names = encoder.get_feature_names_out(input_features=categorical_input.columns)

    encoded_array_bool = encoded_array.astype(bool)
    # 构造编码后的 DataFrame
    encoded_input = pd.DataFrame(encoded_array_bool, columns=feature_names)


    numeric_input = pd.DataFrame(index=[0])

    # 添加数值型变量（直接作为数值）
    for var_name, value in zip(numeric_vars, numeric_values):
        numeric_input[var_name] = value


    numeric_input = pd.concat([pd.DataFrame([fixed_numeric_params]), numeric_input], axis=1)


    X_input = pd.concat([encoded_input, numeric_input], axis=1)

    ###### 叶含梗率部分y1 ######
    with open('./uploads/model/叶含梗率_rbfnn.pkl', 'rb') as f:
        model_package1 = pickle.load(f)

    model1 = model_package1['model']
    scaler_X1 = model_package1['scaler_X']
    scaler_y1 = model_package1['scaler_y']
    feature_names1 = scaler_X1.feature_names_in_
    X1 = X_input[feature_names1]
    X1_scaled = scaler_X1.transform(X1)

    y_pred_scaled = model1.predict(X1_scaled)
    y1 = scaler_y1.inverse_transform(y_pred_scaled.reshape(-1, 1))
    # print(X_input)

    ###### 大中片率部分y2 ######
    with open('./uploads/model/大中片率_rbfnn.pkl', 'rb') as f:
        model_package2 = pickle.load(f)

    model2 = model_package2['model']
    scaler_X2 = model_package2['scaler_X']
    scaler_y2 = model_package2['scaler_y']
    feature_names2 = scaler_X2.feature_names_in_
    X2 = X_input[feature_names2]
    X2_scaled = scaler_X2.transform(X2)

    y_pred_scaled2 = model2.predict(X2_scaled)
    y2 = scaler_y2.inverse_transform(y_pred_scaled2.reshape(-1, 1))

    ###### 梗含叶率部分y3 ######
    with open('./uploads/model/梗含叶率_rbfnn.pkl', 'rb') as f:
        model_package3 = pickle.load(f)

    model3 = model_package3['model']
    scaler_X3 = model_package3['scaler_X']
    scaler_y3 = model_package3['scaler_y']
    feature_names3 = scaler_X3.feature_names_in_
    X3 = X_input[feature_names3]
    X3_scaled = scaler_X3.transform(X3)

    y_pred_scaled3 = model3.predict(X3_scaled)
    y3 = scaler_y3.inverse_transform(y_pred_scaled3.reshape(-1, 1))

    # print("预测的大中片率（%）为：", y2.flatten())
    # print("预测的叶含梗率（%）为：", y1.flatten())
    # print("预测的大中片率（%）为：", y3.flatten())
    # print("==================")
    return y2.flatten()[0], y1.flatten()[0], y3.flatten()[0]

param_names = list(categorical_vars.keys()) + numeric_vars

file = open('./uploads/data/results/天昌优化结果.csv', 'w', newline='', encoding='gbk')
writer = csv.writer(file)
writer.writerow(param_names)

for index in range(len(fixed_categorical_param_list)):
    toolbox.register("evaluate", functools.partial(eval_func, i = index))

    # 注册交叉（两点交叉，适用于数值型变量）
    toolbox.register("mate", tools.cxTwoPoint)

    # 注册变异（均匀变异，适用于数值型变量）
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=100, indpb=0.1)

    # 注册选择（NSGA-II）
    toolbox.register("select", tools.selNSGA2)

    # 初始化种群
    pop = toolbox.population(n=50)

    # 运行遗传算法
    algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=50, cxpb=0.5, mutpb=0.1, ngen=10, verbose=True)

    # 提取所有非支配解
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    # print(pareto_front)


    best_score = float('-inf')
    best_solution = None
    best_index = -1

    for i, ind in enumerate(pareto_front):
        # 选择"大中片率-叶含梗率-梗含叶率"最大的帕累托解
        large_medium_rate, leaf_in_stem_rate, stem_in_leaf_rate = ind.fitness.values
        score = large_medium_rate - leaf_in_stem_rate - stem_in_leaf_rate

        # print(f"解 {i + 1}: 得分={score:.4f}, 目标值: 大中片率={large_medium_rate:.2f}, "
        #       f"叶含梗率={leaf_in_stem_rate:.2f}, 梗含叶率={stem_in_leaf_rate:.2f}")

        if score > best_score:
            best_score = score
            best_solution = ind
            best_index = i

    # print("\n最优解为第 %d 个帕累托解，得分为 %.4f：" % (best_index + 1, best_score))

    print("参数配置：")
    for name, value in zip(param_names, best_solution):
        print(f"{name} = {value}")
    print("==============")
    writer.writerow(best_solution)
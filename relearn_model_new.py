import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 类别特征（保持不变）
categorical_features = ["待加工原烟数据记录_质量属性（稍差、合格、稍强）"]

# 1. 数据加载（包含所有新增特征和标签）
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        numerical_features = [
            "待加工原烟数据记录_水份（%）",
            "待加工原烟数据记录_温度（℃）",
            "待加工原烟数据记录_烟碱含量（%）",
            "待加工原烟数据记录_叶梗比（%）",
            "待加工原烟数据记录_环境温度（℃）",
            "待加工原烟数据记录_环境湿度（%）",
            "铺叶工段数据记录_环境温度（℃）",
            "铺叶工段数据记录_环境湿度（%）",
            "铺叶工段数据记录_流量（kg/h)",
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
            "打叶工段数据记录_环境温度（℃）",
            "打叶工段数据记录_一打_打辊参数（hz）_原始",
            "打叶工段数据记录_二打_打辊参数（hz）_原始",
            "打叶工段数据记录_三打_打辊参数（hz）_原始",
            "打叶工段数据记录_四打_打辊参数（hz）_原始",
            "打叶工段数据记录_风分参数（hz）_一分_原始",
            "打叶工段数据记录_风分参数（hz）_二分_原始",
            "打叶工段数据记录_风分参数（hz）_三分_原始",
            "打叶工段数据记录_风分参数（hz）_四分_原始",
            "打叶工段数据记录_风分参数（hz）_五分_原始",
            "打叶工段数据记录_风分参数（hz）_六分_原始",
            "打叶工段数据记录_风分参数（hz）_七分_原始",
            "打叶工段数据记录_风分参数（hz）_八分_原始",
            "打叶工段数据记录_风分参数（hz）_九分_原始",
            "打叶工段数据记录_风分参数（hz）_十分_原始",
            "打叶工段数据记录_风分参数（hz）_十一分_原始"
        ]

        targets = [
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

        # 检查所有必要列是否存在（特征+标签）
        all_needed_columns = categorical_features + numerical_features + targets
        missing_columns = [col for col in all_needed_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV数据中缺少必要的列: {missing_columns}")

        # 提取特征和标签
        X = df[categorical_features + numerical_features]
        y = df[targets]  # y包含所有标签（15个）

        return X, y, categorical_features, numerical_features, targets

    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise


# 2. 模型构建（无需修改，MultiOutputRegressor支持任意数量标签）
def build_model(categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    gbdt = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbose=0
        )
    )

    model = Pipeline([('preprocessor', preprocessor), ('regressor', gbdt)])
    return model


# 3. 模型训练与评估（核心修改：扩展评估和误差计算）
def train_and_evaluate(X, y, categorical_features, numerical_features, targets, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_model(categorical_features)
    print(f"开始训练多输出模型（共预测{len(targets)}个标签）...")
    model.fit(X_train, y_train)
    print("模型训练完成!")

    y_pred = model.predict(X_test)  # 形状：[样本数, 15]（15个标签）

    # 评估每个标签的预测效果
    print("\n模型评估指标:")
    for i, target_name in enumerate(targets):
        y_test_i = y_test.iloc[:, i]
        y_pred_i = y_pred[:, i]
        mse = mean_squared_error(y_test_i, y_pred_i)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_i, y_pred_i)
        r2 = r2_score(y_test_i, y_pred_i)
        print(f"\n{target_name}的评估:")
        print(f"  MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # 计算每个标签对应的原始特征的基准误差（核心修改：匹配新增标签与原始特征）
    print("\n原始特征与对应标签的基准误差:")
    for target_name in targets:
        # 原始特征名称 = 标签名称 + "_原始"（根据命名规则匹配）
        raw_feature_name = f"{target_name}_原始"
        # 提取测试集中的原始特征值
        raw_feature = X_test[raw_feature_name]
        # 计算该原始特征与标签的误差
        raw_mse = mean_squared_error(y_test[target_name], raw_feature)
        raw_rmse = np.sqrt(raw_mse)
        print(f"{target_name}的原始特征MSE: {raw_mse:.4f}, RMSE: {raw_rmse:.4f}")

    return model, X_test, y_test, y_pred


# 4. 预测函数（无需修改，支持多输出）
def predict_with_model(model, new_data):
    try:
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame(new_data)
        return model.predict(new_data)  # 返回形状：[样本数, 15]
    except Exception as e:
        print(f"预测时出错: {str(e)}")
        return None


# 5. 主函数（展示20个示例预测，包含所有标签）
def main(file_path):
    X, y, categorical_features, numerical_features, targets = load_data(file_path)
    model, X_test, y_test, y_pred = train_and_evaluate(
        X, y, categorical_features, numerical_features, targets
    )

    sample_count = 5
    if len(X_test) > 0:
        actual_count = min(sample_count, len(X_test))
        sample_data = X_test.iloc[:actual_count]
        sample_predictions = predict_with_model(model, sample_data)

        print(f"\n示例预测结果（共{actual_count}个样本，每个样本包含{len(targets)}个标签）:")
        for i in range(actual_count):
            print(f"\n===== 样本 {i + 1} =====")
            for j, target_name in enumerate(targets):
                print(f"{target_name}: 预测值 = {sample_predictions[i, j]:.4f}, 真实值 = {y_test.iloc[i, j]:.4f}")
    else:
        print("\n测试集为空，无法生成示例预测")

    return model


if __name__ == "__main__":
    data_file_path = "data/jsonData.csv"  # 替换为你的CSV路径
    model = main(data_file_path)

    import joblib
    joblib.dump(model, 'model/relearn_model.pkl')
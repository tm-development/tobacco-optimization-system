#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试RBFNN预测逻辑
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict

# 模拟参数优化文件中的分类变量定义
categorical_vars = OrderedDict({
    "回潮工段数据记录_回潮方式": ["半回", "全回"],
    "一润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"],
    "二润工段数据记录_润叶方式（蒸汽、加水、混合）": ["蒸汽", "加水", "混合"]
})

def get_default_categorical_value(param_name):
    """获取分类参数的默认值"""
    defaults = {
        "待加工原烟数据记录_质量属性（稍差、合格、稍强）": "合格",
        "回潮工段数据记录_回潮方式": "半回",
        "一润工段数据记录_润叶方式（蒸汽、加水、混合）": "混合",
        "二润工段数据记录_润叶方式（蒸汽、加水、混合）": "混合"
    }
    return defaults.get(param_name, "合格")

def test_rbfnn_prediction():
    """测试RBFNN预测功能"""
    print("=" * 50)
    print("🧪 开始测试RBFNN预测逻辑")
    print("=" * 50)
    
    # 模拟输入参数 - 使用默认参数和一些变化的参数
    test_parameters_1 = {
        # 分类参数
        "待加工原烟数据记录_质量属性（稍差、合格、稍强）": "合格",
        "回潮工段数据记录_回潮方式": "半回",
        "一润工段数据记录_润叶方式（蒸汽、加水、混合）": "混合",
        "二润工段数据记录_润叶方式（蒸汽、加水、混合）": "混合",
        
        # 数值参数
        "铺叶工段数据记录_流量（kg/h)": 12000,
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
    }
    
    # 测试参数2 - 修改一些数值参数
    test_parameters_2 = test_parameters_1.copy()
    test_parameters_2["打叶工段数据记录_一打_打辊参数（hz）"] = 50  # 从42改为50
    test_parameters_2["打叶工段数据记录_风分参数（hz）_一分"] = 40  # 从33改为40
    
    # 测试参数3 - 修改分类参数
    test_parameters_3 = test_parameters_1.copy()
    test_parameters_3["回潮工段数据记录_回潮方式"] = "全回"  # 从半回改为全回
    
    test_cases = [
        ("基础参数", test_parameters_1),
        ("修改数值参数", test_parameters_2),
        ("修改分类参数", test_parameters_3)
    ]
    
    for case_name, parameters in test_cases:
        print(f"\n🔬 测试用例: {case_name}")
        print("-" * 30)
        
        try:
            result = predict_with_rbfnn_enhanced(parameters)
            print(f"📊 预测结果:")
            print(f"   大中片率: {result.get('big_medium_rate', 'N/A'):.3f}")
            print(f"   叶含梗率: {result.get('leaf_stem_rate', 'N/A'):.3f}")
            print(f"   梗含叶率: {result.get('stem_leaf_rate', 'N/A'):.3f}")
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            import traceback
            traceback.print_exc()

def predict_with_rbfnn_enhanced(parameters):
    """使用RBFNN模型进行预测的增强版本，包含完整的数据预处理"""
    try:
        print(f"🧠 使用RBFNN模型预测，输入参数数量: {len(parameters)}")
        
        # 构建分类参数DataFrame
        categorical_data = {}
        for param_name in ["待加工原烟数据记录_质量属性（稍差、合格、稍强）", "回潮工段数据记录_回潮方式", 
                          "一润工段数据记录_润叶方式（蒸汽、加水、混合）", "二润工段数据记录_润叶方式（蒸汽、加水、混合）"]:
            categorical_data[param_name] = [parameters.get(param_name, get_default_categorical_value(param_name))]
        
        categorical_df = pd.DataFrame(categorical_data)
        print(f"🏷️ 分类参数DataFrame:")
        print(categorical_df)
        
        # One-Hot编码
        encoder = OneHotEncoder(sparse_output=False, categories=[["稍差", "合格", "稍强"], ["半回", "全回"],
                                                                 ["蒸汽", "加水", "混合"], ["蒸汽", "加水", "混合"]])
        encoded_array = encoder.fit_transform(categorical_df)
        feature_names = encoder.get_feature_names_out(input_features=categorical_df.columns)
        encoded_df = pd.DataFrame(encoded_array.astype(bool), columns=feature_names)
        print(f"🔢 One-Hot编码后特征数量: {encoded_df.shape[1]}")
        
        # 构建数值参数DataFrame
        numeric_data = {}
        all_numeric_params = [col for col in parameters.keys() if col not in categorical_vars.keys()]
        
        for param_name in all_numeric_params:
            if param_name in parameters:
                numeric_data[param_name] = [parameters[param_name]]
        
        numeric_df = pd.DataFrame(numeric_data)
        print(f"🔢 数值参数数量: {numeric_df.shape[1]}")
        
        # 合并特征
        X_input = pd.concat([encoded_df, numeric_df], axis=1)
        
        print(f"📊 输入特征维度: {X_input.shape}")
        print(f"📋 特征列名: {list(X_input.columns)[:10]}...") # 只显示前10个
        
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
                
                print(f"   模型期望特征数量: {len(feature_names)}")
                print(f"   输入特征数量: {X_input.shape[1]}")
                
                # 检查特征匹配情况
                missing_features = set(feature_names) - set(X_input.columns)
                extra_features = set(X_input.columns) - set(feature_names)
                
                if missing_features:
                    print(f"   ⚠️ 缺失特征: {list(missing_features)[:5]}...")
                if extra_features:
                    print(f"   ℹ️ 额外特征: {list(extra_features)[:5]}...")
                
                # 确保特征顺序一致
                try:
                    X_aligned = X_input[feature_names]
                    print(f"   ✅ 特征对齐成功")
                except KeyError as e:
                    print(f"   ❌ 特征对齐失败: {e}")
                    continue
                
                X_scaled = scaler_X.transform(X_aligned)
                print(f"   📏 标准化后形状: {X_scaled.shape}")
                
                y_pred_scaled = model.predict(X_scaled)
                print(f"   🎯 模型预测(标准化): {y_pred_scaled}")
                
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
                print(f"   🎯 模型预测(原始尺度): {y_pred}")
                
                predictions[result_key] = float(y_pred.flatten()[0])
                print(f"✅ {metric_name} 预测完成: {predictions[result_key]:.6f}")
            else:
                print(f"⚠️ 模型文件不存在: {model_path}")
                # 提供默认值
                default_values = {'stem_leaf_rate': 0.6, 'big_medium_rate': 78.0, 'leaf_stem_rate': 1.4}
                predictions[result_key] = default_values[result_key]
        
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

if __name__ == "__main__":
    test_rbfnn_prediction()

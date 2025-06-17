from flask import Flask
from app import create_app
import os
import numpy as np
import json
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
app = create_app()

if __name__ == "__main__":
    # 确保上传目录存在
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    data_dir = os.path.join(upload_dir, 'data')
    model_dir = os.path.join(upload_dir, 'models')
    
    print(f"📂 创建上传根目录: {upload_dir}")
    os.makedirs(upload_dir, exist_ok=True)
    print(f"📂 创建数据目录: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    print(f"📂 创建模型目录: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 验证目录是否存在
    print(f"✅ 上传根目录存在: {os.path.exists(upload_dir)}")
    print(f"✅ 数据目录存在: {os.path.exists(data_dir)}")
    print(f"✅ 模型目录存在: {os.path.exists(model_dir)}")
    print(f"📁 当前工作目录: {os.getcwd()}")
    
    print("🚀 启动烟草生产优化系统...")
    print("📊 访问地址: http://127.0.0.1:5000")
    print("🔧 调试模式: 开启 (自动重载: 关闭)")
    
    # 禁用自动重载来避免watchdog问题
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
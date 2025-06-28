# 烟草生产优化系统 (Tobacco Production Optimization System)

基于Flask和机器学习的智能烟草生产参数优化系统，支持多用户权限管理、模型预测和参数优化算法。

## 系统特性

- 🔐 **JWT用户认证**：支持管理员和普通用户两种角色权限控制
- 📊 **数据管理**：支持CSV/Excel文件上传、预览和管理
- 🤖 **模型管理**：支持RBFNN等机器学习模型的上传和管理
- ⚡ **参数优化**：集成粒子群优化(PSO)、遗传算法(GA)等智能优化算法
- 📈 **结果分析**：实时显示优化进度和结果可视化
- 📱 **响应式设计**：现代化暗色主题界面，支持移动端访问

## 用户角色权限

### 管理员 (root)
- 文件管理：上传、删除、预览数据文件
- 模型管理：上传、激活、删除预测模型
- 指标预测：执行单个和批量预测任务
- 参数优化：执行参数优化任务
- 用户管理：管理系统用户

### 普通用户 (user)
- 参数优化：仅可访问参数优化功能

## 项目结构

```
tobacco-optimization-system/
├── app/
│   ├── __init__.py                 # Flask应用初始化
│   ├── routes.py                   # 主路由和蓝图注册
│   ├── models.py                   # 数据模型定义
│   ├── utils.py                    # 工具函数和RBFNN模型
│   └── blueprints/                 # 模块化蓝图
│       ├── __init__.py
│       ├── user_management.py      # 用户管理和JWT认证
│       ├── file_management.py      # 文件上传和管理
│       ├── model_management.py     # 模型管理
│       ├── metric_prediction.py    # 指标预测
│       └── parameter_optimization.py # 参数优化
├── templates/
│   ├── login.html                  # 登录页面
│   ├── index.html                  # 管理员主界面
│   └── user_dashboard.html         # 用户参数优化界面
├── static/
│   ├── css/style.css              # 样式文件
│   ├── js/main.js                 # JavaScript文件
│   └── images/                    # 图像资源
├── uploads/
│   ├── data/                      # 上传的数据文件
│   │   └── results/               # 预测结果文件
│   └── models/                    # 上传的模型文件
├── config.py                      # 应用配置
├── requirements.txt               # Python依赖
├── run.py                         # 应用启动入口
└── README.md                      # 项目文档
```

## 安装和运行

### 环境要求
- Python 3.7+
- Flask 2.3+
- PyJWT 2.8+
- 其他依赖见 requirements.txt

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd tobacco-optimization-system
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动应用**
   ```bash
   python run.py
   ```

4. **访问系统**
   在浏览器中访问 `http://127.0.0.1:5000`

## 默认用户账号

### 管理员账号
- 用户名：`root`
- 密码：`123456`
- 权限：所有功能模块

### 普通用户账号
- 用户名：`user`
- 密码：`123456`
- 权限：仅参数优化模块

## 技术架构

### 后端技术栈
- **Flask**：Web框架
- **PyJWT**：JWT令牌认证
- **Pandas**：数据处理
- **NumPy**：数值计算
- **Scikit-learn**：机器学习
- **Pickle**：模型序列化

### 前端技术栈
- **HTML5/CSS3**：页面结构和样式
- **JavaScript ES6+**：交互逻辑
- **Tailwind CSS**：CSS框架
- **Material Icons**：图标库
- **Chart.js**：图表可视化

### 架构特点
- **模块化设计**：使用Flask蓝图实现功能模块分离
- **JWT认证**：无状态用户认证机制
- **权限控制**：基于装饰器的细粒度权限管理
- **响应式布局**：适配多种设备屏幕

## API接口

### 用户认证
- `POST /api/login` - 用户登录
- `POST /api/logout` - 用户登出
- `GET /api/user/info` - 获取用户信息

### 文件管理 (管理员权限)
- `POST /api/upload_data` - 上传数据文件
- `GET /api/get_uploaded_files` - 获取文件列表
- `DELETE /api/delete_file` - 删除文件

### 模型管理 (管理员权限)
- `POST /api/models/upload` - 上传模型
- `GET /api/models` - 获取模型列表
- `POST /api/models/<filename>/activate` - 激活模型

### 参数优化 (用户/管理员权限)
- `POST /api/optimize_parameters` - 开始参数优化
- `GET /api/optimization_progress/<task_id>` - 获取优化进度
- `GET /api/optimization_history` - 获取优化历史

## 优化算法

系统集成了多种智能优化算法：

1. **粒子群优化 (PSO)**
   - 适用于连续参数优化
   - 收敛速度快，全局搜索能力强

2. **遗传算法 (GA)**
   - 适用于离散和连续参数优化
   - 具有良好的全局搜索能力

3. **模拟退火 (SA)**
   - 适用于复杂目标函数优化
   - 能够跳出局部最优解

4. **梯度下降 (GD)**
   - 适用于可微目标函数
   - 局部搜索效率高

## 部署说明

### 开发环境
```bash
# 设置环境变量
export FLASK_APP=run.py
export FLASK_ENV=development

# 启动应用
flask run
```

### 生产环境
建议使用 Gunicorn + Nginx 部署：

```bash
# 安装Gunicorn
pip install gunicorn

# 启动Gunicorn服务器
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

## 安全配置

### JWT配置
- 默认密钥：`tobacco-optimization-secret-key-2025`
- 令牌有效期：24小时
- 生产环境请更换为强密钥

### 文件上传限制
- 支持格式：CSV, XLSX, XLS, PKL
- 最大文件大小：可在配置中调整
- 文件存储路径：`uploads/` 目录

## 常见问题

### Q: 登录后显示403错误
A: 检查用户权限，确保访问的功能模块符合用户角色权限。

### Q: 文件上传失败
A: 检查文件格式是否支持，文件大小是否超限，uploads目录是否有写权限。

### Q: 优化算法运行缓慢
A: 可以调整算法参数，如减少迭代次数或调整种群大小。

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目维护者：[Your Name]
- 邮箱：[your.email@example.com]
- 项目链接：[https://github.com/yourusername/tobacco-optimization-system]
# 数据库配置说明

本项目现在支持两种数据库：MySQL 和 SQLite

## 数据库类型切换

### 方法一：使用环境变量

在运行应用前设置环境变量：

```bash
# 使用SQLite（默认）
set DATABASE_TYPE=sqlite

# 使用MySQL
set DATABASE_TYPE=mysql
```

### 方法二：使用切换脚本

运行数据库切换脚本：

```bash
python switch_database.py
```

该脚本提供了交互式界面，可以：

- 查看当前数据库配置
- 切换数据库类型
- 测试数据库连接

## SQLite 配置

SQLite 是默认的数据库类型，具有以下优势：

- **无需安装**：Python 内置支持
- **轻量级**：单文件数据库
- **易于部署**：无需额外的数据库服务
- **适合开发**：快速启动和测试

### SQLite 配置参数

```python
DATABASE_TYPE = 'sqlite'
SQLITE_DB_PATH = './tobacco.db'  # 数据库文件路径
```

数据库文件将在项目根目录自动创建。

## MySQL 配置

如果需要使用 MySQL，需要：

1. **安装 MySQL 服务**
2. **安装 Python 依赖**：

   ```bash
   pip install PyMySQL cryptography
   ```

3. **配置连接参数**：
   ```python
   DATABASE_TYPE = 'mysql'
   MYSQL_HOST = 'localhost'
   MYSQL_PORT = '3306'
   MYSQL_USER = 'root'
   MYSQL_PASSWORD = '123456'
   MYSQL_DB = 'tobacco'
   ```

## 数据迁移

### 从 MySQL 迁移到 SQLite

运行迁移脚本：

```bash
python migrate_to_sqlite.py
```

该脚本会：

1. 备份 MySQL 数据到 JSON 文件
2. 在 SQLite 中重建数据库结构
3. 导入备份的数据
4. 验证迁移结果

### 从 SQLite 迁移到 MySQL

1. 首先确保 MySQL 服务运行正常
2. 切换数据库类型为 MySQL
3. 运行应用让其自动创建表结构
4. 手动导入需要的数据

## 依赖安装

### 仅使用 SQLite

```bash
pip install -r requirements-sqlite.txt
```

### 支持两种数据库

```bash
pip install -r requirements.txt
```

## 生产环境建议

- **开发/测试环境**：建议使用 SQLite
- **生产环境**：根据数据量和并发需求选择：
  - 小型应用：SQLite
  - 大型应用：MySQL

## 常见问题

### Q: 如何查看当前使用的数据库类型？

A: 运行 `python switch_database.py` 选择"查看当前数据库配置"

### Q: SQLite 数据库文件在哪里？

A: 默认在项目根目录的 `tobacco.db` 文件

### Q: 如何备份 SQLite 数据？

A: 直接复制 `tobacco.db` 文件即可

### Q: 切换数据库后原来的数据怎么办？

A: 使用 `migrate_to_sqlite.py` 脚本进行数据迁移

### Q: 应用启动时报数据库错误？

A: 检查：

1. 数据库类型配置是否正确
2. MySQL 服务是否运行（如果使用 MySQL）
3. 数据库文件权限（如果使用 SQLite）
4. 相关 Python 包是否安装

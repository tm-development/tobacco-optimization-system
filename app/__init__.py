from flask import Flask
from config import Config
import os
from app.models import db  # 导入数据库对象
import logging
import pymysql
import sqlite3
import os

def create_app():
    # 设置模板和静态文件路径为项目根目录
    app = Flask(__name__,
                template_folder='../templates',
                static_folder='../static')
    app.config.from_object(Config)
    
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 初始化数据库
    db.init_app(app)
    
    # 测试数据库连接
    try:
        with app.app_context():
            if app.config['DATABASE_TYPE'].lower() == 'mysql':
                # MySQL数据库连接测试
                conn = pymysql.connect(
                    host=app.config['MYSQL_HOST'],
                    port=int(app.config['MYSQL_PORT']),
                    user=app.config['MYSQL_USER'],
                    password=app.config['MYSQL_PASSWORD']
                )
                conn.close()
                logger.info("成功连接到MySQL数据库")
                
                # 创建数据库（如果不存在）
                conn = pymysql.connect(
                    host=app.config['MYSQL_HOST'],
                    port=int(app.config['MYSQL_PORT']),
                    user=app.config['MYSQL_USER'],
                    password=app.config['MYSQL_PASSWORD']
                )
                cursor = conn.cursor()
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {app.config['MYSQL_DB']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"确保MySQL数据库 {app.config['MYSQL_DB']} 存在")
            else:
                # SQLite数据库连接测试
                sqlite_path = app.config['SQLITE_DB_PATH']
                db_dir = os.path.dirname(sqlite_path)
                
                # 确保数据库目录存在
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                
                # 测试SQLite连接
                conn = sqlite3.connect(sqlite_path)
                conn.close()
                logger.info(f"成功连接到SQLite数据库: {sqlite_path}")
            
            # 创建数据库表
            db.create_all()
            logger.info("数据库表创建/更新完成")
            
    except Exception as e:
        logger.error(f"数据库连接或初始化失败: {str(e)}")
        # 在生产环境中可能需要采取更多措施，比如退出应用或使用备用数据库
        print(f"⚠️ 数据库错误: {str(e)}")
        print(f"💡 如果使用SQLite，请确保有写入权限。如果使用MySQL，请检查连接配置")
    
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'results'), exist_ok=True)
    
    # 注册蓝图
    from app.routes import register_blueprints
    register_blueprints(app)
    
    return app
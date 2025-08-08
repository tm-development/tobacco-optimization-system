#!/usr/bin/env python3
"""
数据库迁移脚本：从MySQL迁移到SQLite
使用此脚本将MySQL中的数据迁移到SQLite数据库
"""

import os
import sys
import json
from datetime import datetime
import sqlite3
import pymysql
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from app import create_app
from app.models import db, OptimizationResult


def backup_mysql_data():
    """备份MySQL数据到JSON文件"""
    print("📤 开始备份MySQL数据...")
    
    try:
        # 连接MySQL数据库
        conn = pymysql.connect(
            host=Config.MYSQL_HOST,
            port=int(Config.MYSQL_PORT),
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB,
            charset='utf8mb4'
        )
        
        # 查询所有优化结果
        query = "SELECT * FROM optimization_results"
        df = pd.read_sql(query, conn)
        conn.close()
        
        # 转换为JSON格式
        data = df.to_dict('records')
        
        # 处理datetime字段
        for record in data:
            for key, value in record.items():
                if isinstance(value, datetime):
                    record[key] = value.isoformat()
        
        # 保存到JSON文件
        backup_file = f"mysql_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ MySQL数据已备份到: {backup_file}")
        print(f"📊 共备份 {len(data)} 条记录")
        
        return backup_file, data
        
    except Exception as e:
        print(f"❌ 备份MySQL数据失败: {str(e)}")
        return None, None


def restore_to_sqlite(backup_data):
    """将备份数据恢复到SQLite"""
    print("📥 开始恢复数据到SQLite...")
    
    try:
        # 临时设置为SQLite模式
        os.environ['DATABASE_TYPE'] = 'sqlite'
        
        # 创建Flask应用并初始化数据库
        app = create_app()
        
        with app.app_context():
            # 删除现有表并重新创建
            db.drop_all()
            db.create_all()
            
            # 插入数据
            success_count = 0
            for record in backup_data:
                try:
                    # 处理datetime字段
                    created_at = datetime.fromisoformat(record['created_at']) if record.get('created_at') else datetime.now()
                    updated_at = datetime.fromisoformat(record['updated_at']) if record.get('updated_at') else datetime.now()
                    
                    # 创建新记录
                    optimization_result = OptimizationResult(
                        task_id=record['task_id'],
                        input_parameters=record.get('input_parameters'),
                        parameters=record['parameters'],
                        modified_parameters=record.get('modified_parameters'),
                        metrics=record['metrics'],
                        modified_metrics=record.get('modified_metrics'),  # 新字段
                        file_path=record['file_path'],
                        created_at=created_at,
                        updated_at=updated_at,
                        is_modified=record.get('is_modified', False)
                    )
                    
                    db.session.add(optimization_result)
                    success_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 插入记录失败: {record.get('task_id', 'unknown')} - {str(e)}")
                    continue
            
            # 提交事务
            db.session.commit()
            
            print(f"✅ 成功恢复 {success_count}/{len(backup_data)} 条记录到SQLite")
            
    except Exception as e:
        print(f"❌ 恢复数据到SQLite失败: {str(e)}")


def verify_migration():
    """验证迁移结果"""
    print("🔍 验证迁移结果...")
    
    try:
        # 设置为SQLite模式
        os.environ['DATABASE_TYPE'] = 'sqlite'
        
        app = create_app()
        
        with app.app_context():
            # 查询SQLite数据库中的记录数
            count = OptimizationResult.query.count()
            print(f"📊 SQLite数据库中共有 {count} 条记录")
            
            # 显示最近的几条记录
            recent_records = OptimizationResult.query.order_by(OptimizationResult.created_at.desc()).limit(5).all()
            
            print("📝 最近的5条记录:")
            for record in recent_records:
                print(f"  - {record.task_id}: {record.created_at}")
                
    except Exception as e:
        print(f"❌ 验证迁移结果失败: {str(e)}")


def main():
    """主函数"""
    print("🚀 开始数据库迁移：MySQL -> SQLite")
    print("=" * 50)
    
    # 检查是否需要备份MySQL数据
    backup_choice = input("是否备份MySQL数据? (y/n): ").lower().strip()
    
    if backup_choice == 'y':
        backup_file, backup_data = backup_mysql_data()
        
        if backup_data:
            restore_choice = input("是否将备份数据恢复到SQLite? (y/n): ").lower().strip()
            
            if restore_choice == 'y':
                restore_to_sqlite(backup_data)
                verify_migration()
        else:
            print("❌ 备份失败，跳过恢复步骤")
    else:
        print("ℹ️ 跳过MySQL数据备份")
        
        # 直接创建空的SQLite数据库
        os.environ['DATABASE_TYPE'] = 'sqlite'
        app = create_app()
        
        with app.app_context():
            db.create_all()
            print("✅ SQLite数据库已创建")
    
    print("\n🎉 迁移完成！")
    print(f"💡 SQLite数据库文件位置: {Config.SQLITE_DB_PATH}")
    print("💡 要使用SQLite，请设置环境变量: DATABASE_TYPE=sqlite")
    print("💡 要切换回MySQL，请设置环境变量: DATABASE_TYPE=mysql")


if __name__ == '__main__':
    main()

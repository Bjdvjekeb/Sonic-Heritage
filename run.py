#!/usr/bin/env python3
import os
import sys
import logging

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

from app import app

if __name__ == '__main__':
    port = 5501
    try:
        from waitress import serve
        print(f"启动服务在 http://localhost:{port}")
        serve(app, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"启动失败: {e}")
        # 尝试使用Flask内置服务器
        print("尝试使用Flask开发服务器...")
        app.run(host='0.0.0.0', port=port, debug=True)
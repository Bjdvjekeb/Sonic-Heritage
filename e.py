# integrate.py
import subprocess
import sys
import time

def main():
    # 首先运行AI训练
    print("开始训练AI模型...")
    result = subprocess.run([sys.executable, "ai_training_script.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("AI训练失败:")
        print(result.stderr)
        return
    
    print("AI训练完成，启动Web服务...")
    
    # 然后启动Web服务
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    main()
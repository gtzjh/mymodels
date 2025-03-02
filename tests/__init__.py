import os
import sys
import pytest

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests():
    """执行所有测试的入口函数"""
    pytest.main(["-v", "--cov=.", "--cov-report=html"])

if __name__ == "__main__":
    run_tests()

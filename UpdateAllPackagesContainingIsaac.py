import subprocess
import re
import sys

# 获取所有已安装的包
installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=freeze']).decode('utf-8')

# 过滤出名称中包含 "isaac" 的包
isaac_packages = [line.split('==')[0] for line in installed_packages.splitlines() if 'isaac' in line.lower()]

# 更新这些包
for package in isaac_packages:
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', package])
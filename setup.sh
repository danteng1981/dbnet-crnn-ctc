#!/bin/bash

# 企业级智能体平台 - 自动化设置脚本

echo "🚀 开始创建项目文件..."

# 创建目录结构
mkdir -p src/core src/utils src/middleware src/agent src/services src/api/v1
mkdir -p tests monitoring docs

# 创建空的__init__.py文件
touch src/__init__.py
touch src/core/__init__.py
touch src/utils/__init__.py
touch src/middleware/__init__.py
touch src/agent/__init__. py
touch src/services/__init__.py
touch src/api/__init__.py
touch src/api/v1/__init__. py
touch tests/__init__.py

echo "✅ 目录结构创建完成"
echo "📝 请继续手动添加代码文件内容"
echo ""
echo "下一步："
echo "1. 将我提供的每个文件内容复制到对应位置"
echo "2. 运行:  git add ."
echo "3. 运行: git commit -m 'feat: 添加企业级智能体平台'"
echo "4. 运行: git push"

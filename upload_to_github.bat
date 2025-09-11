@echo off
REM 5G网络切片优化项目 - Git上传脚本
echo 🌐 5G网络切片优化项目 - Git上传脚本
echo ===============================================

REM 检查Git是否安装
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git未安装或不在PATH中
    echo 请先安装Git: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo ✅ Git已安装

REM 初始化Git仓库（如果还没有的话）
if not exist .git (
    echo 📂 初始化Git仓库...
    git init
) else (
    echo 📂 Git仓库已存在
)

REM 添加所有文件
echo 📝 添加项目文件...
git add .

REM 提交
echo 💾 提交更改...
git commit -m "Initial commit: 5G Dynamic Network Slicing Optimization System

- 基于Transformer架构的5G动态网络切片优化系统
- 支持eMBB、URLLC、mMTC三种切片类型
- 实时预测和智能资源分配
- 完整的数据处理和可视化模块
- 独立演示版本，无需复杂依赖"

REM 设置分支为main
echo 🌿 设置主分支...
git branch -M main

REM 添加远程仓库
echo 🔗 添加远程仓库...
git remote remove origin 2>nul
git remote add origin https://github.com/runjiezhu/5G-slices-optimizing.git

REM 推送到GitHub
echo 🚀 推送到GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo ✅ 项目成功上传到GitHub!
    echo 🌐 仓库地址: https://github.com/runjiezhu/5G-slices-optimizing
    echo.
    echo 🎯 接下来您可以:
    echo   • 访问GitHub仓库查看项目
    echo   • 克隆到其他机器: git clone https://github.com/runjiezhu/5G-slices-optimizing.git
    echo   • 邀请协作者一起开发
) else (
    echo ❌ 上传失败，请检查:
    echo   • GitHub仓库是否已创建
    echo   • 是否有推送权限
    echo   • 网络连接是否正常
)

echo.
echo 按任意键退出...
pause >nul
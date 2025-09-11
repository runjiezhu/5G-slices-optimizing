# 5G网络切片优化项目 - Git上传脚本 (PowerShell版本)

Write-Host "🌐 5G网络切片优化项目 - Git上传脚本" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# 检查Git是否安装
try {
    git --version | Out-Null
    Write-Host "✅ Git已安装" -ForegroundColor Green
} catch {
    Write-Host "❌ Git未安装或不在PATH中" -ForegroundColor Red
    Write-Host "请先安装Git: https://git-scm.com/download/win" -ForegroundColor Yellow
    Read-Host "按Enter键退出"
    exit 1
}

# 初始化Git仓库（如果还没有的话）
if (-not (Test-Path ".git")) {
    Write-Host "📂 初始化Git仓库..." -ForegroundColor Yellow
    git init
} else {
    Write-Host "📂 Git仓库已存在" -ForegroundColor Green
}

# 添加所有文件
Write-Host "📝 添加项目文件..." -ForegroundColor Yellow
git add .

# 提交
Write-Host "💾 提交更改..." -ForegroundColor Yellow
$commitMessage = @"
Initial commit: 5G Dynamic Network Slicing Optimization System

- 基于Transformer架构的5G动态网络切片优化系统
- 支持eMBB、URLLC、mMTC三种切片类型  
- 实时预测和智能资源分配
- 完整的数据处理和可视化模块
- 独立演示版本，无需复杂依赖
"@

git commit -m $commitMessage

# 设置分支为main
Write-Host "🌿 设置主分支..." -ForegroundColor Yellow
git branch -M main

# 添加远程仓库
Write-Host "🔗 添加远程仓库..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin https://github.com/runjiezhu/5G-slices-optimizing.git

# 推送到GitHub
Write-Host "🚀 推送到GitHub..." -ForegroundColor Yellow
try {
    git push -u origin main
    
    Write-Host "✅ 项目成功上传到GitHub!" -ForegroundColor Green
    Write-Host "🌐 仓库地址: https://github.com/runjiezhu/5G-slices-optimizing" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🎯 接下来您可以:" -ForegroundColor Yellow
    Write-Host "  • 访问GitHub仓库查看项目" -ForegroundColor White
    Write-Host "  • 克隆到其他机器: git clone https://github.com/runjiezhu/5G-slices-optimizing.git" -ForegroundColor White
    Write-Host "  • 邀请协作者一起开发" -ForegroundColor White
    
} catch {
    Write-Host "❌ 上传失败，请检查:" -ForegroundColor Red
    Write-Host "  • GitHub仓库是否已创建" -ForegroundColor Yellow
    Write-Host "  • 是否有推送权限" -ForegroundColor Yellow  
    Write-Host "  • 网络连接是否正常" -ForegroundColor Yellow
    Write-Host "错误信息: $_" -ForegroundColor Red
}

Write-Host ""
Read-Host "按Enter键退出"
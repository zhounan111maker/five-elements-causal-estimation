#!/bin/bash
echo "开始上传五行因果估计研究文件..."
echo "=================================="

cd ~/five-elements-causal-estimation

# 创建目录
mkdir -p papers data src figures

echo "1. 复制论文文件..."
cp ~/Desktop/"五元素因果估计：一种融合五行理论与双重机器学习的新范式.md" papers/ 2>/dev/null || echo "  ⚠️  论文.md文件已复制或不存在"
cp ~/Desktop/"五行论在因果估计效应领域的应用研究.docx" papers/ 2>/dev/null || echo "  ⚠️  论文.docx文件已复制或不存在"

echo "2. 复制数据集..."
cp ~/Desktop/"BankChumers.csv" data/ 2>/dev/null || echo "  ⚠️  数据集文件已复制或不存在"

echo "3. 复制算法代码..."
cp ~/Desktop/main9.py src/ 2>/dev/null || echo "  ⚠️  main9.py文件已复制或不存在"
cp ~/Desktop/main9test2.py src/ 2>/dev/null || echo "  ⚠️  main9test2.py文件已复制或不存在"

echo "4. 复制所有实验结果图片..."
# 所有实验图片
cp ~/Desktop/"交叉验证各折偏差图.png" figures/ 2>/dev/null || echo "  ⚠️  交叉验证各折偏差图.png已复制或不存在"
cp ~/Desktop/"跨域验证效果对比图.png" figures/ 2>/dev/null || echo "  ⚠️  跨域验证效果对比图.png已复制或不存在"
cp ~/Desktop/"屏幕截图2025-11-24 233114.png" figures/ 2>/dev/null || echo "  ⚠️  屏幕截图2025-11-24 233114.png已复制或不存在"
cp ~/Desktop/"特征重要性Top10.png" figures/ 2>/dev/null || echo "  ⚠️  特征重要性Top10.png已复制或不存在"
cp ~/Desktop/"五行论坛维验证—图流.png" figures/ 2>/dev/null || echo "  ⚠️  五行论坛维验证—图流.png已复制或不存在"
cp ~/Desktop/"五行模型结果对比表.png" figures/ 2>/dev/null || echo "  ⚠️  五行模型结果对比表.png已复制或不存在"
cp ~/Desktop/"业务逻辑分组ATE流程图.png" figures/ 2>/dev/null || echo "  ⚠️  业务逻辑分组ATE流程图.png已复制或不存在"
cp ~/Desktop/"噪声鲁棒性对比图.png" figures/ 2>/dev/null || echo "  ⚠️  噪声鲁棒性对比图.png已复制或不存在"

echo "5. 检查文件结构..."
echo "papers/ 文件夹:"
ls -1 papers/ 2>/dev/null || echo "  空文件夹"
echo ""
echo "data/ 文件夹:"
ls -1 data/ 2>/dev/null || echo "  空文件夹"
echo ""
echo "src/ 文件夹:"
ls -1 src/ 2>/dev/null || echo "  空文件夹"
echo ""
echo "figures/ 文件夹:"
ls -1 figures/ 2>/dev/null || echo "  空文件夹"

echo "6. 添加并提交到Git..."
git add .
git commit -m "研究更新: $(date '+%Y-%m-%d %H:%M:%S')

包含:
- 论文文档
- BankChumers数据集
- 算法代码(main9.py, main9test2.py)
- 所有实验图片(交叉验证、特征重要性、五行模型等)"

echo "7. 推送到GitHub私有仓库..."
git push origin main

echo ""
echo "=================================="
echo "上传完成！"
echo ""
echo "访问你的私有仓库:"
echo "https://github.com/zhounan111maker/five-elements-causal-estimation"

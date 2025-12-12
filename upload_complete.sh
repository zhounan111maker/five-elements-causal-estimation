#!/bin/bash
echo "开始上传五行因果估计研究文件（修正版）..."
echo "========================================"

cd ~/five-elements-causal-estimation

# 创建目录
mkdir -p papers data src figures

echo "1. 复制论文文件..."
cp ~/Desktop/"五元素因果估计：一种融合五行理论与双重机器学习的新范式.md" papers/
cp ~/Desktop/"五行论在因果估计效应领域的应用研究.docx" papers/

echo "2. 复制数据集..."
# 注意：数据集文件名是"Bankchurner figures"，类型是XLS工作表
if [ -f ~/Desktop/"Bankchurner figures" ]; then
    cp ~/Desktop/"Bankchurner figures" data/
    echo "  已复制数据集: Bankchurner figures"
else
    # 如果文件名不同，尝试其他可能的名字
    cp ~/Desktop/BankChumers.csv data/ 2>/dev/null || echo "  ⚠️ 未找到数据集文件"
fi

echo "3. 复制算法代码..."
cp ~/Desktop/main9.py src/
cp ~/Desktop/main9test2.py src/

echo "4. 复制所有实验结果图片..."
# 图片在桌面的figures文件夹内
if [ -d ~/Desktop/figures ]; then
    cp ~/Desktop/figures/*.png figures/
    echo "  从figures文件夹复制了图片"
else
    # 如果不在figures文件夹，尝试从桌面根目录复制
    cp ~/Desktop/*.png figures/ 2>/dev/null || echo "  ⚠️ 未找到图片文件"
fi

echo "5. 检查文件结构..."
echo "papers/ 文件夹:"
ls -1 papers/
echo ""
echo "data/ 文件夹:"
ls -1 data/ 2>/dev/null || echo "  空文件夹"
echo ""
echo "src/ 文件夹:"
ls -1 src/
echo ""
echo "figures/ 文件夹:"
ls -1 figures/ 2>/dev/null || echo "  空文件夹"

echo "6. 添加并提交到Git..."
git add .
git commit -m "上传完整研究材料（修正版）

包含:
- 论文文档（Markdown和Word版）
- Bankchurner figures数据集（44MB，XLS格式）
- 算法代码：main9.py和main9test2.py
- 所有实验图片：交叉验证、特征重要性等"

echo "7. 推送到GitHub私有仓库..."
echo "注意：数据集文件较大（44MB），上传可能需要一些时间..."
git push origin main

echo ""
echo "========================================"
echo "上传完成！"
echo ""
echo "访问你的私有仓库:"
echo "https://github.com/zhounan111maker/five-elements-causal-estimation"

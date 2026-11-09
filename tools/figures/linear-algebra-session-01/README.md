# 第一节笔记配图

四张图分别解释行的交点、列向量组合、共线向量的可达集合，以及习题 1.1 的共面关系。数学来源为笔记中的具体方程，全部是确定性的 TikZ 绘图；图 4 是平面关系的斜投影，不表示实际长度与角度。

依赖：Tectonic、Python 3 和 PyMuPDF（`fitz`）。在仓库根目录运行：

```sh
python3 tools/figures/linear-algebra-session-01/build_figures.py \
  --output /tmp/linear-algebra-session-01 \
  --assets source/images/courses/gilbert-linear-algebra/session-01
```

输出包含 PDF、SVG 和 PNG 预览；网站采用 SVG，以免放大后公式标签模糊。图中的数学标签用 LaTeX，中文解释放在网页图注中。修改后需查看实际图件，检查标签、裁切及箭头方向。图和笔记均按 CC BY-NC-SA 4.0 许可分享。

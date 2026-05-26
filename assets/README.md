# CrystalLM Assets

本目录包含CrystalLM项目的图片资源。

## 需要创建的图片

### 1. system_overview.png
**描述**: CrystalLM系统架构总览图

**内容要求**:
- 显示完整的双向翻译流程：CIF → HSSR → T5 Model → Natural Language
- 包含三个主要组件：
  - HSSR Encoder（将CIF转换为HSSR）
  - T5 Translator（双向翻译模型）
  - Constrained Decoder（确保有效的HSSR输出）
- 使用箭头表示数据流向
- 标注关键指标（BLEU-4: 41.2, EM: 31.5%）

**建议尺寸**: 1200×600 pixels, 300 DPI
**格式**: PNG

---

### 2. hssr_comparison.png
**描述**: HSSR格式与CIF格式对比图

**内容要求**:
- 三列布局：
  - 左列：CIF格式示例（500+ tokens）
  - 中列：HSSR格式示例（100-200 tokens）
  - 右列：自然语言描述
- 使用MOF-5作为示例
- 突出显示token数量差异
- 使用颜色编码区分不同部分

**建议尺寸**: 1400×800 pixels, 300 DPI
**格式**: PNG

---

### 3. translation_examples.png
**描述**: 翻译示例展示

**内容要求**:
- 展示两个完整的翻译案例：
  - MOF-5: S2T翻译
  - UiO-66: T2S翻译
- 包含输入和输出
- 使用不同颜色区分S2T和T2S
- 标注关键信息字段

**建议尺寸**: 1200×900 pixels, 300 DPI
**格式**: PNG

---

### 4. performance_scaling.png
**描述**: 性能-数据规模曲线图

**内容要求**:
- X轴：数据集大小（2K, 4K, 6K, 8K样本）
- Y轴：BLEU-4分数（0-50）
- 两条曲线：
  - T5-base（蓝色）
  - T5-large（红色）
- 标注最终性能点
- 添加图例和网格线

**建议尺寸**: 800×600 pixels, 300 DPI
**格式**: PNG

---

### 5. error_distribution.png
**描述**: 错误类型分布图

**内容要求**:
- 两个饼图并排：
  - 左：S2T错误分布
    - Semantic errors (45%)
    - Factual errors (30%)
    - Grammatical errors (15%)
    - Other (10%)
  - 右：T2S错误分布
    - Format errors (40%)
    - Value errors (35%)
    - Missing fields (20%)
    - Other (5%)
- 使用不同颜色
- 添加百分比标签

**建议尺寸**: 1000×500 pixels, 300 DPI
**格式**: PNG

---

### 6. attention_visualization.png
**描述**: 注意力权重可视化

**内容要求**:
- 热图显示HSSR tokens和生成文本之间的注意力权重
- X轴：HSSR tokens（[META], [COMP], [LATT], etc.）
- Y轴：生成的文本单词
- 使用颜色渐变表示注意力强度（蓝色→红色）
- 添加颜色条图例

**建议尺寸**: 1000×800 pixels, 300 DPI
**格式**: PNG

---

## 图片制作工具建议

### Python/Matplotlib
适合制作：performance_scaling.png, error_distribution.png

```python
import matplotlib.pyplot as plt
import numpy as np

# 示例代码
fig, ax = plt.subplots(figsize=(10, 6))
# 添加绘图代码
plt.savefig('performance_scaling.png', dpi=300, bbox_inches='tight')
```

### 绘图工具
适合制作：system_overview.png, hssr_comparison.png, translation_examples.png

推荐工具：
- **Figma**: 专业UI设计工具
- **draw.io**: 免费流程图工具
- **PowerPoint**: 快速原型制作
- **Adobe Illustrator**: 专业矢量图形

### 热图工具
适合制作：attention_visualization.png

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 示例代码
plt.figure(figsize=(12, 10))
sns.heatmap(attention_matrix, cmap='RdYlBu_r', 
            xticklabels=hssr_tokens, 
            yticklabels=text_words)
plt.savefig('attention_visualization.png', dpi=300, bbox_inches='tight')
```

---

## 使用说明

1. 创建图片后，将其放置在此目录下
2. 确保文件名与上述规格一致
3. 在README.md中引用图片：`![描述](assets/图片名.png)`
4. 提交前检查所有图片是否正确显示

---

## 注意事项

- 所有图片应使用高分辨率（300 DPI）
- 使用PNG格式以保证质量
- 保持一致的配色方案
- 确保文字清晰可读
- 图片大小应适中（每个文件<2MB）

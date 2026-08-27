# 教学插图说明

`01_quantization_map.png` 采用三行并列布局，分别呈现 W8A16、W4A16 和 FP8 KV Cache；每一行都从高精度张量、经过量化码与 scale 元数据，再回到计算近似值，适合用来区分“压缩对象”和“尺度粒度”。

`02_gptq_awq_concept.png` 以同一校准 batch 分出左右两条路径。左侧 GPTQ 用列序量化、二阶几何等高线和向后传播的残差箭头表达层输出重构与误差补偿；右侧 AWQ 用激活柱状图、显著通道和缩放旋钮表达激活感知的等价缩放。图中浮点保护路径仅作为教学对照，正文会明确说明真实 AWQ 通常采用缩放而非硬件不友好的混合精度权重。

三张图均为原创教学概念图，不代表某一实现的精确内核、数据格式位级编码或性能测量。正文将把真实的定量比较交给程序生成的实验图表与可复现 JSON 结果。

`03_fp8_kv_cache_groups.png` 从左到右显示 token 增长、K/V 存储、单个 attention head 向量的 per-tensor / per-head / group scale 层次，以及在 attention read 前的恢复路径。它用于解释 scale 粒度选择，不把图中示例字节码视为实际 E4M3/E5M2 编码。

`04_weight_quantization_tradeoffs.png` 已通过视觉核验。左侧柱图显示在实验的“理想码值 + FP32 scale”账本下，不同权重量化方案相对 FP16 的逻辑存储量；右侧以对数纵轴显示固定合成评估 batch 的线性层输出 MSE。图中明确标注结果来自固定合成张量，不能作为模型或硬件 benchmark。

`05_kv_quantization_tradeoffs.png` 已通过视觉核验。左侧图对比 FP16、每向量一个 scale 与每 16 个元素一个 scale 的逻辑 KV 存储占比；右侧以对数纵轴同时展示 KV 张量 MSE 与微型 attention 输出 MSE。图中可见更细 group scale 以额外 scale 元数据换取更低误差，且标题与角注已表明这是确定性教学实验而非 serving benchmark。

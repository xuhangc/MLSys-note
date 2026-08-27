# 视觉资产核验说明

- `01_paged_kv_memory_concept.png`：从左到右表现不同长度的请求 token 流、各请求独立的逻辑块表、非连续的 GPU 物理 KV 页池，以及底部按逻辑顺序恢复的上下文。颜色对应请求，虚线空位表示未占用或尾块未满。
- `02_radix_prefix_reuse_concept.png`：中央亮色主干表示多个请求共享的前缀；新请求先沿共享主干匹配，再在黄橙色后缀处分叉。右侧多分支各自映射到 KV 块，表示一次计算、多请求复用。
- `03_chunked_prefill_scheduler_concept.png`：已核验。左侧长 token 序列被切为四段紫色 prefill chunk；中央为调度器与有限 token budget；绿色时钟符号表示需优先维持的 decode 进度；右侧缓存柜以雪花、盾牌和回收符号区分冷缓存、受保护热前缀和驱逐路径。

这些为原创教学概念图；文中将以图注说明，不将其作为精确实现图或性能数据图。

# 教学图与图表说明

`01_profiling_optimization_loop.png` 用七个闭环节点表达性能优化的实验协议：先以 contract 固定可比较的工作负载，再依次获得 baseline、trace、瓶颈假设、单变量改动与复测，最后以证据包进入决策闸门。图中的“same workload”强调任何结论都依赖对比口径恒定。

`02_end_to_end_timeline_bottlenecks.png` 将 CPU 与 GPU 时间线并排。条纹空白段、数据/传输、计算和同步 callout 共同说明，GPU 空闲可能来自数据等待或宿主同步；因此热点函数/内核本身不一定就是端到端瓶颈。下方 working-set 曲线为概念示意，不对应本实验的 MB 数值。

`03_evidence_to_decision.png` 从 workload manifest、metrics、profiler trace 和 quality guard 汇聚为 evidence packet，并导向 ACCEPT、TUNE、REJECT。它用于解释为什么“变快”必须同时满足质量、尾延迟与证据要求。

`04_stage_breakdown.png` 与 `05_candidate_tradeoffs.png` 均来自 `code/profiling_lab.py` 的实际 CPU 墙钟测量。前者显示 baseline 与被选方案的数据等待阶段差异；后者用 P95 与明确声明的 working-set proxy 呈现单变量候选的权衡。候选标签经错开与引导线处理后可读；图中 working set 并非 allocator 或 GPU peak memory 采样。

`06_latency_distribution.png` 由同一测量样本绘制 ECDF，用于将平均值与尾延迟放入同一分布视角。所有定量图都标注为 controlled CPU teaching workload，不能解释为真实模型或 GPU 的性能 benchmark。

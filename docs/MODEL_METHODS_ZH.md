# WMH–GM SuStaIn：冻结方法、参数与结果解释

本文描述仓库当前正式默认配置及实现行为。正式参数来源是 `config/analysis.yaml`；运行时保存的配置、输入身份和源码哈希共同确定一次分析。合成验证使用单独的 profile、参数及目录，不能作为患者结果。

## 一、研究范围与数据角色

- **LS：健康参考。** 只用年龄在参考支持范围内的健康记录估计年龄、性别与真实颅内容积相关的均值及残差尺度。
- **C3：发现集。** 主分析限制于 50–75 岁，确定亚型数量、事件顺序、亚型比例及不确定性。
- **GB：外部应用集。** 主分析同样限制于 50–75 岁，使用冻结的 LS 参考和 C3 模型进行分型分期，不参与规范化、阈值调整、选 K 或种子选择。
- **全龄应用：范围敏感性。** C3/GB 全龄表使用相同参考和模型。参考范围之外的个体标记为 `outside_reference_range`；不单独重拟合模型，不把外推结果等同于年龄范围内验证。
- 原始输入为20个WMH区域负荷和20个纯GM体积。最终模型为WMH19＋GM19，共38维：排除LS全零的`wmh_basal_ganglia_layer4_ml`，将原GM17纹状体–苍白球和原GM19基底前脑体积相加后作为一个GM指标。丘脑GM16保持独立。合并发生在log及LS标准化之前。原始40维、合并来源、排除理由及LS证据均冻结保存；C3/GB不参与分区方案或参考参数估计。自动处理资格与数值QC决定入模，人工审核仅保留记录，不加入站点效应和临床结局模块。

年龄边界默认 50 和 75，**两端均包含**。实际边界从 `reference.age_min` 与 `reference.age_max` 读取。50–75 岁参考符合本项目的目标年龄，但不构成全年龄常模。GM/WMH 事件顺序是横断面数据支持的模型顺序；stage 不是病程年数，不能据此直接推断个体的真实时间轨迹或因果关系。

## 二、成熟工具及固定版本

推断核心为 [UCL POND 官方 pySuStaIn](https://github.com/ucl-pond/pySuStaIn)，使用 `ZscoreSustain`，不重写或修改官方 EM/MCMC。

| 项目 | 固定来源 |
|---|---|
| pySuStaIn | `708fa22d89c9692dbeafcc2c31a1c8460ced5640`，2026-03-30；包声明版本仍为 0.1 |
| awkde | `1c31e55fe54c0cad80ab423a9605fc9ddfb2614c` |
| kde_ebm | `3ad8b648a4a2d0a8df0707f382b54c4ebef0805c` |
| `AbstractSustain.py` SHA256 | `5d0719ea48f04377215ee8c128b4483b7a055411a92d7f3989fa1e10a51b93a3` |
| `ZscoreSustain.py` SHA256 | `4c510a7e10332eea7361f1fe7ff7c8d66961843bf6993850c22b0027eda9ef66` |

安装使用仓库的小型 wheel 与锁定的 PyPI 依赖。即使本地 wheel 安装不带 Git 的 `direct_url`，运行入口仍核对上述模块字节哈希。目标工作站不需要下载 MRI、深度学习权重或运行时网络图谱。

方法引用：

1. Young AL et al. *Uncovering the heterogeneity and temporal complexity of neurodegenerative diseases with Subtype and Stage Inference*. Nature Communications, 2018. [doi:10.1038/s41467-018-05892-0](https://doi.org/10.1038/s41467-018-05892-0)
2. Aksman LM et al. *pySuStaIn: a Python implementation of the Subtype and Stage Inference algorithm*. SoftwareX, 2021. [doi:10.1016/j.softx.2021.100811](https://doi.org/10.1016/j.softx.2021.100811)

## 三、LS 独立参考与输入数值检查

每个入模特征分别拟合 OLS：

```text
WMH response = log1p(WMH volume in mL)
GM response  = log(GM volume in mL)

response ~ intercept + natural_cubic_age(df=4, centered)
                     + sex + log(true ICV in mL)
```

使用中心约束避免样条常数方向与截距重复。样条节点、约束、性别编码、系数和残差尺度由 LS 学习并冻结，不能在 C3/GB 重新构建设计基底。GM 取负向标准化残差，使数值越大表示越明显的体积减少；WMH 取正向残差。负 z 值保留，不统一截成零。

| 配置键（reference） | 默认 | 含义 |
|---|---:|---|
| age_min / age_max | 50 / 75 | 主分析与参考年龄支持边界，包含两端 |
| folds | 5 | LS 留出校准诊断折数 |
| seed | 20261018 | LS 折分随机种子 |
| spline_df | 4 | 年龄自然三次样条自由度 |
| min_reference_n | 40 | 参考模型最小样本数的工程门槛 |
| min_residual_sd | 0.00000001 | 数值近零残差尺度的拒绝门槛，不代表生物学界限 |
| sex_encoding | female=0, male=1 | 固定编码；未知/缺失编码不能自动代入 |
| wmh_transform / gm_transform | log1p / log | WMH 保留真实零；GM 必须大于零 |
| covariates | age_spline, sex, log_icv | 不包含站点效应 |
| diagnostic_z_thresholds | 1, 2, 3 | 描述参考残差及异常事件覆盖 |
| diagnostic_coverage_z | 1.96 | 留出诊断覆盖区间 |

LS 五折是**诊断**：每折仅在训练 LS 中拟合均值和残差尺度，检查留出 LS；最后以全 LS 重拟合并冻结。不能用留出数据再次缩放后，把被强制得到的均值零和 SD 一当作独立校准成功。

真实 ICV 不可由非脑室组织总体积替代。零 WMH 不当作缺失，不能人为添加病灶体积。零/近零参考残差 SD、设计矩阵秩不足、非有限 z 或无有效似然支持会阻止正式建模；正态性检验显著或某个 z 阈值覆盖少只形成诊断，不依据全 C3/GB 自动删除特征或事件。需要改变变换、特征或阈值时，应使用明确的新配置和输出版本。

## 四、所有默认建模与资源参数

| 配置键（model） | 默认 | 实际行为 |
|---|---:|---|
| k_max | 6 | C3 候选 K=1…6；K=1 同样是有效结果 |
| folds | 10 | C3 受试者互斥的打乱 KFold；不按站点分层 |
| startpoints | 25 | 每个官方层次模型的 EM 多起点数 |
| cv_iterations | 100000 | 每折、每个候选 K 的正式 MCMC 链长度 |
| fit_iterations | 1000000 | 预先指定主种子的全 C3 拟合链长度 |
| stability_iterations | 100000 | 每个额外种子的全 C3 稳定性链长度 |
| seed | 20261018 | 主种子标识及 C3 折分种子；不是按 GB 结果选出的种子 |
| stability_seeds | 20261019, 20261020 | 两个独立稳定性种子；不替换预先指定主解 |
| z_vals | 1, 2, 3 | 每个特征的异常事件阈值；正式38维产生114个事件 |
| z_max | 5 | 轨迹末端锚定值，必须大于最大事件阈值 |
| posterior_samples | 1000 | 用于个体概率的确定性保留样本上限 |
| display_samples | 1000 | 用于重画 PVD/轨迹的展示样本上限 |
| posterior_batch_size | 128 | 个体推断批大小；不会把患者重复写入结果 |
| burn_in_fraction | 0.1 | 仅用于本项目派生后验和展示抽样；官方 CVIC 保持上游抽样规则 |
| workers | 4 | 外层 CV 折进程数上限；同时受可用内存和 CPU 数限制 |

| 配置键（runtime） | 默认 | 实际行为 |
|---|---:|---|
| profile | formal | 正式模式检查配置与冻结模型特征合同；当前38维，原始输入仍须40维 |
| max_memory_fraction | 0.5 | 当前可用/受 cgroup 限制内存的模型预算比例 |
| memory_safety_factor | 2.0 | 对似然张量、候选位置张量、链数组估算的安全倍数 |
| disk_safety_factor | 1.2 | 全链磁盘空间估算安全倍数；断点已有链扣除后估计新增空间 |
| memory_limit_gb | null | 可另设更小的内存上限；不是把总内存强制扩大 |
| blas_threads | 1 | 每个进程的 BLAS 线程上限 |

**并行范围：** 官方 `use_parallel_startpoints=False`，避免上游 `cpu_count()` 默认开满全部核心；只在外层并行不同 CV 折，最多 4 个，内存不足时自动降至更少进程。全数据主拟合和稳定性种子依次执行。

**实际成本：** 官方每个模型在正式MCMC之前，固定执行三轮、每轮10,000次提议尺度调优。合成测试没有缩短这三轮。拟合选定K仍需层次模型1…K，并为每个层次模型保留完整链。因此主种子1m加两个种子各100k，不是只生成三条选定K的链。以38特征、114事件、K=6、1m次为例，单个K=6的序列float64数组即约5.47 GB，尚未包含其他模型与中间数组。

## 五、CVIC 选择、稳定性与断点

每个 C3 受试者只进入一个测试折。官方对其所属训练折模型的 MCMC 样本计算混合似然；CVIC 按以下顺序汇总：先对模型样本平均该受试者的似然，再取对数，最后对受试者求和并乘 -2。

```text
CVIC(K) = -2 * sum_subject log(mean_MCMC p(x_subject | training_fold_model_K))
```

选择有限 CVIC 的最小值；数值完全并列取较小 K。若 K=6 最优，结果标记 `boundary_optimum`，不能称已确认全局最佳亚型数。训练似然、临床结果和 GB 分配均不用于选 K。折级 `log_likelihood` 是对各 MCMC 样本的测试集对数似然求平均，与 CVIC 的“平均似然后取 log”是不同摘要。

选定 K 后：

1. 预先指定主种子拟合全 C3，保留 1m 正式链。
2. 两个额外种子各拟合 100k，作为稳定性分析。
3. 使用事件平均位置的 Kendall tau 和 Hungarian 一对一匹配，将不同种子及折的亚型对应到主模型。
4. 输出每个匹配的顺序相关、亚型比例差及个体原始亚型分配的 ARI。报告其数值，不用任意统一阈值把“完成运算”等同于“稳定性已证实”。

上游 pickle 不保存完整随机数发生器的恢复状态。为使断点跳过已完成模型不会改变后续随机流，本项目**逐 K 调用原官方接口**，每个模型种子由 `SeedSequence([run_seed, fold+1, K])` 确定；全数据拟合使用 `fold=-1`。实际种子写入 `model/cv/folds.json` 与 `model/fit/model_seeds.json`。已完成的层次模型从官方 pickle 读取；没有改变分裂、EM、调优或 MCMC 公式。

缓存身份包括：独立 LS `reference_hash`、C3 实际矩阵哈希、特征字典、配置、官方源码与项目包装源码哈希，以及 Python/NumPy/SciPy 等实际推断依赖版本。C3 训练身份不引用 GB 快照；GB 转换/验证单独保存全输入转换身份。拟合前重新核对 CV 关键参数与最小 CVIC 选择，不能在改变事件阈值后复用旧选型；仅改变最终链或稳定性参数不要求重新选择 K。参数或身份不符时拒绝复用，要求新输出目录。不完整 pickle 改名保留为 `.incomplete.*` 证据后，只补算缺失模型。CV 每折有独立锁与完成标记。

## 六、概率语义、单例兼容与标签

### 6.1 官方输出原样保留

官方完整链和原始个体输出保留在 `pickle_files`。本项目的 `posterior_*.npz` 另保留所选保留样本上的七项 `official_*` 数组：

```text
official_ml_subtype, official_prob_ml_subtype,
official_ml_stage, official_prob_ml_stage,
official_prob_subtype, official_prob_stage,
official_prob_subtype_stage
```

固定版本的 `_calculate_likelihood()` 返回的联合阶段数组没有乘亚型比例 f，而官方 `prob_subtype_stage` 路径直接归一化该返回项；另两个边际概率路径已经乘 f。因此不能把官方联合数组的边际直接等同于官方另外两个概率数组。官方 `prob_ml_stage` 也不是经过重新归一化的“给定所选亚型的分期置信度”。[固定源码](https://github.com/ucl-pond/pySuStaIn/blob/708fa22d89c9692dbeafcc2c31a1c8460ced5640/pySuStaIn/AbstractSustain.py)

### 6.2 派生联合后验

保持官方 EM/MCMC 和阶段似然不变，使用每个保留样本的亚型比例计算一致的联合后验。令 L 是官方阶段似然（已经包含阶段先验），f 为同次 MCMC 的亚型比例：

```text
q_i,stage,subtype(t) = f_subtype(t) * L_i,stage,subtype(t)
                      / sum_all_stage_and_subtype[f(t) * L(t)]

posterior_joint = mean_t q(t)
posterior_subtype = sum_stage posterior_joint
posterior_stage = sum_subtype posterior_joint
```

**必须先逐样本归一化，再平均后验。** 先平均似然后统一归一化是另一种计算，不能代替此处的参数不确定性积分。程序拒绝全零/非有限似然，不通过 epsilon 制造概率支持。测试确认派生两个边际与使用相同保留样本的官方两个边际一致。

亚型取 `posterior_subtype` 最大值；阶段取给定该亚型后的条件阶段概率最大值。`stage_conditional_probability` 已按所选亚型归一化；同时保留完整联合、两个边际、期望阶段、亚型熵及 `stage_zero_probability`。

### 6.3 单例和亚型编号

固定版官方单例推断存在 `squeeze/reshape` 缺陷。包装只在该官方推断调用内部临时把一行复制为两行，再截回原始行数；患者概率相互独立，因此这不改变原例概率。训练、患者数量、保存表格均不重复记录。真实官方测试比较单例、整批与尾批结果，确认一致。

官方/模型原始下标从 0 开始；展示亚型按用于后验的保留链平均比例降序编号 1…K。`subtype_order` 保存原始到展示的顺序。`selected_model.npz` 的链保持原始顺序，绘图须应用 `subtype_order`；`cv_display.npz` 已按主模型顺序匹配，其 `subtype_order` 为恒等排列。

若给定亚型的最可能 stage 为 0，展示标签为 `Undifferentiated`，展示 `subtype=0`；`subtype_raw_zero_based`、原始模型输出和完整概率仍保留。Stage 0 是本模型的早期/未分化状态，不等同于已证明无疾病或影像完全正常。

## 七、完整链、展示样本与正式交付

| 位置 | 内容与用途 |
|---|---|
| `model/cv/fold_<i>/pickle_files/` | 每折所有候选 K 的完整官方链及 EM 解；用于审计与重新计算 |
| `model/cv/cvic.tsv` / `fold_loglikelihood.tsv` | 选型与逐折验证似然 |
| `model/fit/primary/pickle_files/` | 主种子 1…选定 K 的全部官方链 |
| `model/fit/seed_<seed>/pickle_files/` | 稳定性种子的全部官方链 |
| `model/fit/selected_model.npz` | 确定性抽样的展示链、事件阈值、特征标签、原始样本索引和排序映射 |
| `model/fit/cv_display.npz` / `cv_stability.tsv` | 匹配到主模型的跨折展示链及稳定性摘要 |
| `model/fit/stability.tsv` | 两个额外种子与主模型的匹配结果 |
| `model/fit/assignments_C3_*.tsv` | C3 主分析及全龄冻结应用结果 |
| `validation/assignments_GB_*.tsv` | GB 主分析及全龄冻结应用结果 |
| `posterior_*.npz` | 全体对应个体的官方及派生概率，不只保存最大概率标签 |

展示抽样从去掉配置烧入比例后的链均匀取不重复索引，数量最多为 `display_samples`/`posterior_samples`。展示序列以整数无损保存；个体概率以 float64 压缩保存。**没有删除或截断官方完整链。** 完整训练归档与无需 MRI/训练环境即可离线重画的展示包用途不同；展示包不是继续长链 MCMC 的检查点。

`runs/synthetic_model_validation`仅验证真实官方模型输出到报告的技术接口，含3个合成特征、20个C3和5个GB模拟记录。它不执行正式LS参考拟合，不验证当前正式38维、六个K、10折或正式长链分析。新增展示夹具必须明确标注synthetic display-only；该目录及图表不能混入正式结果。

完整技术示例由`config/synthetic.yaml`和`scripts/synthetic_integration.py`生成，输出`runs/synthetic_validation`。它从真实生成的三维体积数据开始：LS主参考100人，C3主发现40人，GB主外部应用12人，各另含两名范围外模拟记录。实际执行statsmodels LS参考、五折留出诊断、官方候选K=1–2的两折SuStaIn、主种子和两个稳定性种子、冻结外部应用、报告、展示包及独立重建验收。其短链和三维合同只适用于软件验证，不放宽生产原始40维／模型38维合同；没有用假参考系数或后补假校准曲线代替真实计算。

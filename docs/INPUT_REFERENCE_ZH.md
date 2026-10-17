# 输入导入与LS冻结常模

本模块从已经完成的分割结果提取原始体积，构建LS健康参考，再对C3发现集和GB验证集应用同一个冻结变换。它不重新分割患者MRI，不使用临床结局，也不拟合站点效应。正式特征为WMH20加纯GM20。

## 1. 工作站路径与配置入口

所有路径及研究参数写在`config/analysis.yaml`。在目标工作站复制`config/local.example.yaml`为`config/local.yaml`，通过`extends: analysis.yaml`仅覆盖实际需要调整的值。相对路径以配置解析得到的`project_dir`为根，与执行命令时的终端目录无关。

本地开发项目位于`/home/zhenzong2/analysis/Substain/sustain_analysis`。其父项目仅保留TMS015、TMS016历史技术演示，不能替代LS/C3/GB正式数据。下面是已有工作流记录形成的工作站默认路径；这些路径需在目标机器上用`doctor`核实。

| 队列 | 分割衍生目录 | 原始名册 |
|---|---|---|
| LS | `/mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_LS/derivatives` | `/data02/linzhenzong/Substain_GB/config/participants_LS.tsv` |
| C3 | `/data/usersdir/linzhenzong/Substain/derivatives/substain_features` | `/data/usersdir/linzhenzong/Substain/config/participants.tsv` |
| GB | `/mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_GB/derivatives/substain_features` | `/data02/linzhenzong/Substain_GB/config/participants_gb.tsv` |

例如：

```yaml
extends: analysis.yaml
project_dir: ..
output_dir: runs/formal_v1
cohorts:
  LS:
    derivatives: /actual/LS/derivatives
    participants: /actual/LS/config/participants_LS.tsv
    metadata: null
  C3:
    input_bundle: /actual/compact/C3
  GB:
    derivatives: /actual/GB/derivatives/substain_features
    participants: /actual/GB/config/participants_gb.tsv
    metadata: null
```

`metadata: null`表示名册本身已含年龄和性别。若明确配置了metadata路径而文件不存在，导入会报错，不会忽略这一缺失。metadata与名册按`participant_id`一对一合并，只填补缺失的age/sex；`60`与`60.0`、`female`与`F`视为同值，真实冲突会报错。

### 输入相关配置键

| 配置键 | 默认值或含义 |
|---|---|
| `project_dir` | 项目根目录；相对路径以配置文件位置解析 |
| `output_dir` | 本次不可混用输入的输出目录，默认`runs/formal_v1` |
| `paths.mapping` | `resources/mappings/muse_gm_macro20_v1.tsv` |
| `cohorts.<LS/C3/GB>.derivatives` | 队列衍生根目录，包含`sub-<participant_id>/status/` |
| `cohorts.<name>.participants` | TSV名册，ID必须非空、唯一，保留前导零 |
| `cohorts.<name>.metadata` | 可选TSV；默认`null` |
| `cohorts.<name>.input_bundle` | 可选轻量包；设置后优先从包导入，不访问原分割目录 |
| `cohorts.<name>.manual_qc` | 可选人工QC TSV；未设置时在衍生目录`tables/`寻找原有审核表 |
| `cohorts.<name>.source_root` | 当`feature_json`是相对路径时的解析根；未设置时取participants父目录的父目录 |
| `cohorts.<name>.path_prefix_map` | 显式旧路径前缀→新路径前缀映射；只匹配完整目录前缀，用于分割结果搬迁后的指针修正 |
| `ingest.required_stages` | `skullstrip, registration, lesion, wmh_seg, wmh, t1, qc`；不能省略这七阶段 |
| `ingest.age_valid_min` | `0`；年龄必须严格大于它 |
| `ingest.age_valid_max` | `120`；年龄必须不大于它 |
| `ingest.icv_consistency_rtol` | `0.000001`；ICV mL与mm³一致性相对容差 |
| `ingest.sex_mapping` | 默认接受male/female、M/F、男/女；数值性别编码须由数据负责人在配置中明确含义 |
| `ingest.manual_qc_required` | `false`；本项目按已批准规格仅记录人工QC，不以它门控入模 |

路径前缀映射示例：

```yaml
cohorts:
  C3:
    source_root: /new/Substain
    path_prefix_map:
      /old/Substain: /new/Substain
```

## 2. 导入资格与纯GM20

每例必须满足下列条件：

1. 名册包含有效年龄及可明确解释的性别。
2. 七阶段JSON均存在、可读、身份一致，且`status`全部为`pass`。`cleanup`完成本身不代表分析成功。
3. `wmh`与`t1`状态内的`details.feature_json`均指向可读、身份一致的JSON。
4. WMH的`raw_ml`必须完整包含20个固定特征，数值有限且非负。
5. `gm119_ml`包含完整119个原生标签；字段整体缺失时允许读取`native145_ml`中的同一119标签。单个GM标签可以为0，但聚合后的每个GM宏区必须大于0。
6. `dlicv_icv.available`必须为true，且`icv_label702_ml`有限、严格大于0。若也有mm³，二者须一致。非脑室组织体积不能替代真实ICV。
7. 读取期间源文件不得变化。输出不能落入源衍生目录或覆盖源文件。

个体缺文件、无权限、失败阶段、身份不符、非有限值、缺少GM标签和ICV等均写入逐例审计；其余合格病例继续导出。不把失败者补零，不使用旧常模z值补缺失。名册重复ID、配置错误或两份metadata矛盾属于整批输入契约错误，需要修正后再运行。

人工`pass/fail/unreviewed`原值保留。原始理由、审核者及审核hash随压缩JSON保存。它们不决定本轮自动入模资格；导入也不把旧人工记录重新认证为当前图像的审核结论。

### 两个20维特征块

WMH特征顺序为基底节、额、枕、颞、顶，每区1–4层。单位统一为mL。特征名沿用`wmh_<region>_layer<1..4>_ml`。

纯GM20由新mapping对119个官方GM原生标签求体积和。旧macro20含122标签；本版仅将第19组中的35脑干、61/62腹侧间脑排除，保留75/76左右基底前脑并命名`basal_forebrain`。其余19组不变。20组互斥，合计覆盖119个GM标签，每组均非空。特征名为`gm_<macro_id>_ml`。

这一步只需已有原始体积JSON，不需要重跑患者分割。WMH继承上游病灶对侧替代的处理结果；T1原始体积没有病灶剔除或镜像处理，不能将其表述为病灶排除后的GM指标。

## 3. 轻量转移包

在保存分割结果的机器执行：

```bash
python -m substain_sustain export-inputs --config config/local.yaml --cohort C3 --destination /actual/compact/C3
```

新目录必须不存在，避免覆盖旧快照。包包含：

- `cohort.tsv`：合格病例元数据、真实ICV及40个原始特征。
- `audit.tsv`：全部名册病例的纳入状态与原因。
- `roster.tsv`：原名册及补充年龄/性别。
- `raw_records.jsonl.gz`：实际读取的原始状态、feature JSON、人工QC记录和源hash。
- `feature_dictionary.tsv`：固定特征顺序、名称、模态与atlas标签。
- `manifest.json`：队列身份、mapping hash、每个文件SHA256和源证据。

目标工作站将该队列配置为`input_bundle`后执行`ingest`即可。重导入不会依赖JSON中原机器的绝对路径；它重新检查包hash、队列与受试者身份、七阶段状态、JSON重聚合体积/ICV，以及表中age/sex与原名册的一致性。

包不含个体MRI、深度学习模型权重和分割环境。它含受试者资料，按研究数据管理要求保留在受限存储，不能推送到代码仓库。

## 4. LS自然样条参考

主分析支持范围为50–75岁，含两端。LS中该范围内自动合格者训练常模；C3与GB不参与参数学习。范围外个体仍保留在全年龄敏感性表，但不进入`*_primary_z.tsv`。

默认模型对每个特征调用`statsmodels.api.OLS`：

```text
WMH: y = log(1 + volume_mL)
GM:  y = log(volume_mL)
E[y] = intercept + natural_cubic_spline(age, df=4, centered)
       + sex + log(ICV_mL)
sigma = sqrt(SSE / (n - rank))
WMH_z = (y - E[y]) / sigma
GM_z  = (E[y] - y) / sigma
```

因此WMH增加与GM减少均对应更高的疾病方向z。该z是相对于LS条件参考的偏离，不自动构成病理诊断或风险预测。

Patsy自然样条的内部节点取LS训练年龄分位点。df=4时使用3个内部节点；边界固定为配置的50与75岁。中心约束只从相应训练集估计，再与截距一起组成满秩设计。所有节点、边界、中心约束、系数、残差SD和特征顺序写入JSON；预测时不重新估计它们，也不pickle不支持序列化的`DesignInfo`。

自然样条在边界外延续线性尾部。全年龄表的`age_in_support=false`明确标记这种外推。LS50–75岁覆盖符合本项目研究年龄定位，但它不证明全年龄常模有效；外推结果与主分析分别展示。

### 常模相关配置键

| 配置键 | 默认值与用途 |
|---|---|
| `reference.age_min`, `age_max` | `50`, `75`；LS训练与主分析支持范围 |
| `reference.folds` | `5`；LS内部OOF诊断折数 |
| `reference.seed` | `20261018`；性别分层随机分折 |
| `reference.spline_df` | `4`；中心约束后年龄样条自由度 |
| `reference.min_reference_n` | `40`；最低计算人数门槛，不代表样本量充分性证明 |
| `reference.min_residual_sd` | `1e-8`；任一特征残差SD过低即停止，不自动删特征 |
| `reference.sex_encoding` | `{female: 0, male: 1}` |
| `reference.wmh_transform` | `log1p`；本版已验证的WMH变换 |
| `reference.gm_transform` | `log`；本版已验证的GM变换 |
| `reference.covariates` | `[age_spline, sex, log_icv]`；不接受站点协变量 |
| `reference.diagnostic_z_thresholds` | `[1, 2, 3]`；输出超过各阈值的比例 |
| `reference.diagnostic_coverage_z` | `1.96`；输出落在对应对称z范围内的比例；改变它时需按实际阈值解释覆盖率 |

每折独立估计年龄节点、中心约束、OLS系数和残差SD；被留出的LS样本不参与该折拟合。保存折分配、逐折模型和OOF z，可追溯检查泄漏。诊断包括z均值/SD、覆盖率、阈值比例，以及z与年龄、性别、log(ICV)的相关。`LS_fit`是训练内诊断；正式查看常模泛化校准时应使用`LS_oof`。

年龄分布不足以建立节点、设计不满秩、任一性别无法满足分折、残差SD过低等会明确失败，不静默改df、删协变量或换方法。应先检查数据和诊断，再以明确的新配置与新输出目录开展方法敏感性分析。

## 5. 执行顺序、产物与冻结身份

```bash
python -m substain_sustain doctor --config config/local.yaml
python -m substain_sustain ingest --config config/local.yaml
python -m substain_sustain reference --config config/local.yaml
```

`inputs/`保留每队列的原始体积表、原始名册、逐例审计、原始JSON压缩记录、特征字典和输入manifest。`subject_key`使用`cohort:participant_id`，防止不同队列同名ID碰撞。

`reference/`保留：

| 文件 | 内容 |
|---|---|
| `model.json` | 全LS支持内训练得到的冻结参数、算法身份和参考hash |
| `LS_z.tsv`, `C3_z.tsv`, `GB_z.tsv` | 全部合格年龄的z及`age_in_support`标记 |
| `LS_primary_z.tsv`, `C3_primary_z.tsv`, `GB_primary_z.tsv` | 50–75岁主分析表 |
| `LS_oof_z.tsv` | LS内部留出预测，含fold编号 |
| `fold_assignments.tsv`, `fold_models.json` | 分折与各折独立参考参数 |
| `diagnostics.tsv` | `LS_fit`、`LS_oof`及每折的逐特征诊断 |
| `reference_training_audit.tsv` | LS年龄支持范围的纳入/排除记录 |
| `manifest.json` | 文件hash、样本数、参数、参考身份和转换快照身份 |

三个身份分开使用：

- `reference_hash`只依赖支持范围内实际LS训练数据、特征字典、常模配置及算法源码/依赖身份。仅改变GB不会改变它，也不会改变`model.json`。
- `input_snapshot_hash`记录本次LS/C3/GB输入快照，用于转换与完整数据追溯。
- `transform_hash`结合冻结参考身份与输入快照身份，保护已经生成的三队列z表。

同一输出目录发现任何输入或参数变化会拒绝覆盖。仅GB改变时，参考身份不变，但转换身份改变，仍需新输出目录。后续C3模型选择与拟合的训练身份应使用`reference_hash`和C3训练矩阵，不能把GB快照身份当作训练信息。

全部结果展示还需保留下游SuStaIn模型样本、CV结果、GB冻结推断、绘图代码和空间展示atlas；本节输入/常模文件是其中的必要组成部分。ROI模型脑图不需要转移每个患者的NIfTI，但不能将ROI数值投影解释为新完成的个体或群体体素统计。

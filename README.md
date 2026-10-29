# LS–C3–GB WMH/GM SuStaIn

从已经完成的WMH和GM分割结果，建立LS健康参考、C3发现模型和GB冻结验证。原始输入保留WMH20＋纯GM20；主模型使用50–75岁参与者、WMH19＋纯GM19，共38维。WMH排除LS全零的`wmh_basal_ganglia_layer4_ml`；GM将原17纹状体–苍白球与原19基底前脑的mL体积相加，保留原16丘脑独立，再进行log和LS参考标准化。自动质控通过即可入模，不进行站点校正。所有科研参数和输入路径均在 `config/`；模块按功能分区并有中文注释。

**软件已提供完整分析与报告入口。仓库不含患者数据，技术测试不代表真实队列已完成建模。**

## 工作站安装

目标环境：Linux x86_64、glibc ≥ 2.31、已有Python 3.10。原分割项目的T1环境提供Python，分析依赖装入独立 `.venv`，不会改动原环境。安装只需GitHub与pip源；不需要apt、conda、Hugging Face、MRI模型下载或C++编译器。

```bash
git clone https://github.com/zhenzonglin/Substain-SuStaIn.git
cd Substain-SuStaIn
python3 scripts/bootstrap.py --config config/install.json
cp config/local.example.yaml config/local.yaml
```

安装器默认使用工作站现有的pip软件源、代理和证书配置（`pip_index_url: null`），不强制访问pypi.org。
如需单次指定源，可加 `--index-url https://你的已有软件源/simple`；也可在安装配置填写该地址。
如果遇到 `Network is unreachable` 后接 `No matching distribution`，先处理软件源连通性，不要据此更改锁定版本或删除虚拟环境。
修正后运行 `git pull --ff-only` 并重试原安装命令即可复用已建环境。

已有完整离线依赖时可使用 `--wheelhouse /完整wheel目录`。仓库的 `vendor/wheels` 仅包含三个上游工具，不是完整依赖目录。

若原Python位置不同，修改 `config/install.json` 的 `base_python`，或使用：

```bash
python3 scripts/bootstrap.py --config config/install.json --python /实际位置/envs/t1/bin/python
```

编辑 `config/local.yaml`，检查三队列的derivatives、participants与输出目录。参与者表已有age/sex时，metadata保持null；如需补充元数据，填写真实存在的TSV路径。不要把输出目录设在任何源数据目录内。

安装展示资源并检查环境：

```bash
.venv/bin/python scripts/install_display_assets.py --config config/local.yaml
bash scripts/run.sh doctor --config config/local.yaml
```

WMH图谱优先读取已有分割项目；缺失时从配置指定的GitHub固定提交下载约0.8 MB。GM图谱和公开背景已随仓库提供，工作站无需再次分割模板。全部输入及展示资源就绪后，分析和报告无需联网。

## 正式执行

需要从约30个计算进程增加到60时，先按[SMT容量检查与可选切换](docs/SMT60_ZH.md)只读预检，无需更新正在运行的代码。60个逻辑CPU不等于60物理核；125 GiB机器在原正式内存预算下可能无法放行60个EM起点，资源不足时保留旧任务继续运行。该入口保留完整K链，并复用校验通过的完整起点。

若切换预检提示“缓存留出似然无效”，且检查发现原始似然为0，请使用本版本的[数值下溢修正与只读诊断](docs/NUMERICAL_UNDERFLOW_ZH.md)。修正从冻结链重算对数域CVIC与个体概率，保留原始零值和缓存，不重跑K=1，不修改官方EM/MCMC；同时报告训练下限可能造成的影响。

已使用分块加速版本、目前每折已有K=1缓存的工作站，可立即切换受限起点并行：

```bash
cd /data02/linzhenzong/Substain-SuStaIn &&
git pull --ff-only &&
.venv/bin/python scripts/switch_startpoints.py --config config/local.38.yaml --apply
```

该入口不等待K=2，先严格核验，再保留导入、LS参考及所有完整模型缓存，精准停止旧CV并后台续跑。资源上限为30物理核对应60逻辑CPU，每折最多3个EM起点，实际活跃折数由原内存预算限制；38维和全部科学参数不变。新版保存完整起点检查点。参数、保留边界、预检与监控见[起点并行切换说明](docs/STARTPOINT_PARALLEL_ZH.md)。

正式计算默认采用32人分块似然，并记录起点、EM轮和MCMC迭代评估的真实进度。科学规格、官方EM/MCMC和固定提交不变。旧版CV缓存须经安全切换迁移：

```bash
git pull --ff-only
.venv/bin/python scripts/switch_acceleration.py --config config/local.38.yaml --apply
```

该命令先在工作站全部C3输入上核验逐元素一致性，备份配置和缓存身份，再精准停止旧CV及其后台续跑器并自动重启。输入、LS参考及完整缓存保留；未保存的EM/MCMC需重跑。已进入fit时拒绝自动切换。[完整切换、监控与备份说明](docs/ACCELERATION_ZH.md)。

默认主规格：38维、114个事件、z阈值1/2/3、K=1–6、C3十折、25起点、CV每模型10万次MCMC、主模型100万次、两个独立稳定性种子各10万次。`feature_selection`明确记录排除依据，每次仍验证LS主年龄范围内该特征全零；其他特征退化仍会停止。原始40维不删列，GB不参与特征选择。完整官方后验可能较大，运行前检查剩余磁盘与配置的内存限额。

```bash
bash scripts/run.sh ingest --config config/local.yaml
bash scripts/run.sh reference --config config/local.yaml
bash scripts/run.sh crossvalidate --config config/local.yaml
bash scripts/run.sh fit --config config/local.yaml
bash scripts/run.sh validate --config config/local.yaml
bash scripts/run.sh report --config config/local.yaml
bash scripts/run.sh bundle --config config/local.yaml
```

也可用 `bash scripts/run.sh all --config config/local.yaml` 连续执行ingest至report；展示包独立生成。每阶段保存状态、配置快照和错误日志。中断后重跑相同阶段会复用完整且身份一致的产物；中断的MCMC链重新计算，不宣称链内续跑。改变分析规格请设置新的 `output_dir`。

展示包默认在配置 `report.bundle_dir` 指向的目录。验收：

```bash
bash scripts/run.sh verify-bundle --bundle /实际路径/sustain-display-bundle.tar.gz
```

该命令在独立临时目录使用包内代码、数值源和图谱重新绘制，并核对图源表哈希、PNG像素及TIFF尺寸/DPI；不依赖原MRI、原项目目录或网络。

## 无sudo的独立CV计时诊断

长时间没有首个缓存且无法读取调用栈时，可在另一终端运行：

```bash
.venv/bin/python scripts/probe_cv.py --config config/local.38.yaml
```

参数集中在`config/cv_probe.yaml`。探针核对冻结身份，使用原fold 0的全部C3训练数据、K=1原种子和原特征，额外启动一个低优先级单线程进程，最多测一轮EM或10分钟；每5秒显示完成的似然计算数。一轮完成后显示真实耗时、候选logL变化和顺序变化。只写`.cache/cv_probe/`，不停止旧任务、不改正式参数、官方库或缓存。探针会增加少量资源竞争。

**独立探针的推进不能证明旧worker没有空转，也不能给出完整CV的剩余时间。** 它用于检验同一数据下官方计算能否完成及耗时是否很长。具体证据边界与输出说明见[CV进度诊断](docs/CV_PROGRESS_ZH.md)。

## 正式分析监控

启用acceleration.progress_enabled后，显示当前K、优化批次、起点、EM轮、事件/候选位置、最近logL及MCMC迭代评估。心跳时间与完成的数值调用数分别记录；旧PID和延迟心跳不会被当作正常推进。

已完成导入和LS参考、当前处于交叉验证时，可将原4并行安全切换为10并行：

以下入口只适用于缓存已经使用当前源码、仅需改变并行数的情况。旧版本CV升级到分块和EM心跳，应使用上方switch_acceleration.py的跨版本切换入口。

```bash
git pull --ff-only
.venv/bin/python scripts/switch_parallel.py --config config/local.38.yaml --apply
```

`config/parallel10.yaml`集中设置10个CV任务、80%可用内存预算、停止等待时间、审计/日志路径和续跑阶段。脚本先核验源码、依赖、输入、LS参考、分折、随机种子与全部科学参数，再检查CPU/内存/磁盘。资源不足10并行时在停止旧任务前退出。它只停止当前项目和配置对应的CV进程，备份原配置及缓存身份，迁移workers与内存预算字段，并后台依次执行crossvalidate、fit、validate、report、bundle、verify-bundle。导入和LS参考不重跑；完整模型/折缓存保留，中断的未保存MCMC链重新计算。已进入fit或后续阶段会拒绝切换。

不加`--apply`只预检。运行中的进程池不会因编辑YAML而扩容；不要直接改workers后同时再开一个任务。主模型及稳定性种子仍依次执行；启用起点并行后各模型内部的独立EM起点可并行。监控同时显示配置并行上限和最近实际分配值；全新任务默认请求10折并行，内存不足时核心调度会降低并发以遵守预算。

在另一终端运行，`--config`必须使用正式启动时的同一份配置：

```bash
.venv/bin/python scripts/monitor.py --config config/local.38.yaml
```

每5秒显示八个阶段的完成、运行、失败、待运行或中断状态；同时显示C3逐折完成情况、主种子/稳定性种子的缓存写出情况、分析进程树CPU和RSS内存，以及整机CPU和可用内存。CPU合计100%代表一个逻辑核心；RSS合计可能重复计算共享页。按Ctrl+C只退出监控。只打印一次可加`--once`。

监控读取`execution/*.json`、折完成标记和Linux文件锁，并匹配实际分析入口及子进程；不依赖日志或旧PID文件，不启动或停止分析。无匹配活进程的残留`running`显示为中断。`config/monitor.yaml`单独配置刷新间隔、采样时间和显示行数，修改它不会改变模型缓存身份。

缓存文件数不等于通过验收的模型数；监控不加载MCMC大文件，也不推算链内进度。旧版`verify-bundle`仅在成功时写验收回执，没有失败状态记录，因此无有效回执且无活进程时显示“未验收”，不能自动区分未启动和失败。监控不是模型产物完整性验收的替代品。

若`reference`报告“LS残差SD低于设定阈值”，先执行只读诊断：

```bash
.venv/bin/python scripts/diagnose_reference.py --config config/local.38.yaml
```

该命令核对已导入输入的哈希，使用同一自然样条和OLS检查全体合格LS及五折训练子集，并汇总问题特征在三队列中的零值/非零值数量。它还对照保存的LS原始WMH JSON快照与导入表，输出均为汇总数字。诊断参数在`config/reference_diagnostics.yaml`；命令不修改输入、参考参数或状态，不启动SuStaIn。

全零、常量和仅某个训练折退化需要分别处理。若LS没有可估计的残差尺度，当前LS残差z定义无法用于该特征；不能通过把SD阈值改为零、手工加小数或借用C3/GB的SD继续正式拟合。先核实数据，再明确记录任何特征或参考方法的变更。

仅检查导入人数和排除原因，可加`--inputs-only`。原始状态标记pass并不保证迁移后原路径仍然可读。

## 手动迁移后的C3与38维配置

已有`config/local.yaml`的工作站，可复制`config/formal38_migrated.example.yaml`为`config/local.38.yaml`；该文件继承LS/GB及其他本机设置，C3指向已确认的`/mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_C3/derivatives/substain_features`，结果使用新的`runs/formal_LS_C3_GB_38_v1`目录。

最终GM19编号：1–16保持原编号；新17=原17+原19；18仍为海马–杏仁核；新19=原20小脑皮层–蚓部。脑干与ventralDC均不加入。不能复用旧40维、39维或此前“丘脑+纹状体”草稿的参考/CV结果。完整更新、启动和监控命令见[38维正式部署](docs/FORMAL38_ZH.md)。

若git pull提示本地config/analysis.yaml会被覆盖，使用部署文档中的upgrade_formal38.py流程。它先保存实际生效的本机配置和原文件，再恢复模板并拉取；更新后的38维配置继承保存快照，因此保留写在analysis.yaml中的本机路径/参数。不要直接restore或reset后丢掉这些设置。

`cohorts.C3.relocate_feature_pointers: true`保留状态指针中`sub-本例ID/`之后的完整相对路径，在配置的derivatives内定位对应文件。即使旧路径仍存在，也使用新副本。它不会遍历猜测文件名，不修改原始JSON；指针越界、ID不一致、实际文件缺失或权限不足继续记录排除。原路径、实际读取路径和文件哈希一并保存。真实阶段失败仍不能入模。

更换配置后重新运行`ingest`只会读取已有分割结果，不重新分割。检查C3导入人数和排除原因后，再从`reference`继续；空队列现在会在参考拟合前明确停止。

## 无需搬运原始MRI

在能访问源队列的工作站导出：

```bash
bash scripts/run.sh export-inputs --config config/local.yaml --cohort C3 --destination /可写目录/C3_compact
```

只转移该小型包，在目标配置中设置 `cohorts.C3.input_bundle`。包包含名册、原始特征JSON、GM119/GM20、WMH20、真实ICV、QC与排除审计及哈希；不含MRI或深度学习权重。LS、GB使用相同入口。患者数据包不能上传GitHub。

## 正式结果保留

- `inputs/`：完整导入清单、合格特征、排除原因和来源快照。
- `reference/`：LS参考参数、OOF诊断、38维模型字典、排除与合并清单、GM119→GM19映射、三队列38维原始体积，以及冻结变换后的全龄/主分析矩阵。
- `model/cv/`：分折、CVIC、逐折似然、全部官方模型和序列一致性。
- `model/fit/`：C3主模型、稳定性复核、完整MCMC链、展示抽样与个体概率。
- `validation/`：冻结GB主分析及全年龄应用结果。
- `report/`：离线HTML、PNG、SVG、600-DPI TIFF、逐图TSV及完整性清单。
- `execution/`：本次运行配置、每阶段状态和失败原因。

上述路径均位于配置的 `output_dir`。完整分析档案须留在工作站；轻量展示包保留全部个体概率与固定绘图样本，不替代完整MCMC档案。原始分割结果继续保留，工程不会删除它们。

## 方法与解释边界

核心为UCL官方[pySuStaIn](https://github.com/ucl-pond/pySuStaIn)，锁定 `708fa22d89c9692dbeafcc2c31a1c8460ced5640`；不重新实现EM/MCMC。LS参考为statsmodels OLS＋Patsy中心约束自然样条。旧Chung/GenMIND z不进入新模型，非脑室组织体积不能替代真实ICV。

GM20由GM119重聚合；第19区仅保留基底前脑。WMH已有急性病灶相关处理；GM未做病灶扣除。人工未审核状态不等于人工质量批准。Stage 0称为“未分化”，阶段不代表病程年数。全年龄分支只是冻结模型的年龄外推敏感性应用。

详细说明：

- [输入、路径与LS参考](docs/INPUT_REFERENCE_ZH.md)
- [模型、概率语义与全部参数](docs/MODEL_METHODS_ZH.md)
- [结果展示与离线重画](docs/REPORT_BUNDLE_ZH.md)
- [配置使用说明](docs/CONFIG_ZH.md)
- [验证记录](docs/VERIFICATION_ZH.md)
- [第三方来源](THIRD_PARTY.md)

## 本地验证

```bash
.venv/bin/python -m pytest -q --basetemp=.cache/pytest
SUBSTAIN_RUN_OFFICIAL_INTEGRATION=1 .venv/bin/python -m pytest tests/test_model.py -q --basetemp=.cache/pytest-official
.venv/bin/python scripts/synthetic_integration.py --config config/synthetic.yaml
```

合成测试明确标记SYNTHETIC；其参数、样本量和输出位置单独配置，不覆盖正式结果。

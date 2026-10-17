# SuStaIn 结果展示与可移动重画包

报告的输入是保存的特征矩阵、LS参考模型、C3模型及GB冻模推断结果。报告不重新分割影像，不访问在线图谱、CDN、网络字体或JavaScript。所有路径使用统一配置加载器解析；相对路径以配置的 `project_dir` 为准。

## 生成正式报告

```bash
substain-sustain report --config config/workstation.yaml
```

报告位于配置 `output_dir` 下的 `report/index.html`。每幅图都有PNG、SVG、600DPI TIFF及同名TSV数值表；格式、DPI、大小、颜色、后验抽样上限与展示阶段均在 `report` 配置区。默认覆盖：

- LS/C3/GB纳入流程与实际排除记录；自动计算完成与人工QC记录分别保留。
- LS折外参考分布、均值、方差、覆盖率及残余协变量相关。
- C3交叉验证CVIC和逐折留出似然。
- 每个亚型的完整事件位置后验概率，以及匹配后的跨折PVD。
- 独立随机种子及跨折匹配后的序列稳定性。
- 基于公开模板的WMH和GM宏区阶段图；颜色是模型隐含区域z值，不是体素统计。
- C3与冻模GB的亚型/阶段分布、个人分配不确定性。
- 50–75岁LS支持范围内主分析与范围外外推的分层描述。范围外分数不代表已验证的全年龄常模。

stage 0显示为“Undifferentiated”；阶段不解释为疾病年限或进展速率。缺少必要参考、模型、跨折结果或图谱时，程序写入 `report/summary.json` 的 `incomplete` 状态并失败，不用空图或模拟图补齐。`runtime.profile=synthetic` 的报告和归档明确标注技术演示，不能作为队列发现。

## 公开展示图谱

`assets` 配置包含WMH20图谱、GM20图谱、公开背景及SHA256清单。仅需在开发端对公开模板进行一次NiChart DLMUSE处理。患者影像不进入展示资源。

准备好配置中的公开模板分割后执行：

```bash
.venv/bin/python scripts/prepare_display_atlas.py --config config/local.yaml
```

脚本从 `assets.source_gm_segmentation` 读取原生DLMUSE标签，按 `paths.mapping` 映射到1–20宏区，同时复制 `source_wmh_atlas` 与 `source_background`，写出清单与来源哈希。它不启动深度学习推断；不同内容的已有目标不会被覆盖。GM宏区包含15个官方皮质聚合和5个项目定义非皮质聚合，清单保留该来源说明。

## 建包与真实重画验收

```bash
substain-sustain bundle --config config/workstation.yaml --destination bundles/formal_display.tar.gz
substain-sustain verify-bundle --bundle bundles/formal_display.tar.gz
```

建包前必须已经生成完整报告。相同包名不覆盖。包内包含：

- 轻量特征、队列审核、参考参数、LS折外结果、特征字典、模型选择及稳定性表。
- 保存的展示后验样本、冻模C3/GB分配和完整个人联合亚型–阶段概率。
- 三个公开空间资源及哈希清单。
- 实际使用的重画代码、固定版本显示依赖、配置和重现合同。

包内**不包含已渲染图片**。验收先校验归档每个文件的SHA256，再在新的临时目录仅用包内代码、数值源和公开图谱重新生成全部图表。重画Python进程显式拒绝DNS解析和socket连接，先通过四项受控连接拒绝自检；不改变工作站系统网络配置。验收逐图检查PNG像素、TIFF尺寸/DPI，逐表检查数值TSV哈希，并写出同名 `.verification.json`，其中 `network_disabled=true` 记录该进程级限制。原MRI、原项目目录和原报告图片均不参与重画。

接收端只需Python及pip依赖：

```bash
tar -xzf formal_display.tar.gz
cd sustain-display
python -m pip install -r requirements-report.txt
python rebuild_report.py
```

重新生成的报告为 `data/report/index.html`；安装依赖之后，重画过程不联网。

## 工作站必须永久保留的内容

可移动展示包足以重画报告，但完整方法学复核还需要工作站分析目录。请保留：全部C3交叉验证折模型、最终模型与独立种子的官方 `pickle_files/` 完整MCMC链、模型选择记录、参考模型、完整个人概率、输入快照/审核、配置/软件版本及执行日志。不能把“展示包重画通过”解释为MCMC充分混合、分割质量或疾病结论已经验证。

600DPI TIFF和完整链可能较大；无需通过GitHub传输。GitHub仅交付代码、配置、公开图谱及安装依赖指纹。患者特征和推断结果通过私有数据位置或轻量数据包传入，不提交Git仓库。

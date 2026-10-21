# 配置使用说明

## 文件与覆盖顺序

`config/analysis.yaml` 是完整正式参数文件，`config/local.yaml` 是工作站覆盖文件。`extends: analysis.yaml` 做递归字典合并；列表整体替换。相对路径统一以配置的 `project_dir` 为根，不跟随终端当前目录。`config/local*.yaml/json` 不提交Git。

`config/install.json` 控制Python、虚拟环境、锁文件与pip入口；`config/synthetic.yaml` 控制独立技术验证。官方源代码身份由 `vendor/manifest.json` 锁定，wheel和模块源码校验不匹配时必须停止，不静默升级。

安装来源配置：

- `pip_index_url: null`：沿用当前用户/系统pip配置及`PIP_CONFIG_FILE`、`PIP_INDEX_URL`等环境变量，保留已有代理、额外源和证书。
- `pip_index_url: "https://已有镜像/simple"`：显式指定主索引；命令行`--index-url`优先于配置。
- `wheelhouse: null`：默认不使用离线目录；设为完整wheel目录或传入`--wheelhouse`后，使用`--no-index`并隔离远程软件源配置。该目录必须包含pip、setuptools、wheel及requirements.lock内的全部依赖。
- `pip_timeout_seconds`和`pip_retries`：控制联网重试；设为null则使用pip现有设置。

如果软件源只保存在某个旧虚拟环境的pip.conf，而非用户或系统配置，可通过`PIP_CONFIG_FILE`指定那个配置文件，或用`--index-url`传入已有源地址。安装失败保留已有环境；现有环境Python版本不符时明确停止，不删除或覆盖。

## 主要分区

| 配置区 | 控制内容 |
|---|---|
| project_dir / output_dir | 工程根和本次正式结果目录 |
| paths.mapping | 纯GM20固定映射表 |
| paths.model_mapping | 最终GM119→GM19映射；运行时校验与合并配置一致 |
| cohorts | 每队列derivatives、participants、可选metadata或input_bundle |
| ingest | 自动阶段要求、年龄合法范围、性别映射、ICV一致性 |
| feature_selection | 保留原始40维；排除WMH4，合并原GM17+GM19；expected_model_features=38 |
| reference | LS年龄范围、样条自由度、五折、随机种子、残差SD与诊断阈值 |
| model | K范围、十折、起点、MCMC次数、种子、事件阈值、后验/展示抽样 |
| runtime | formal/synthetic身份、BLAS线程、并行及内存/磁盘安全系数 |
| assets | 展示图谱、背景、冻结清单、已有WMH来源、模板工具与生成参数 |
| report | 图像格式、尺寸、600DPI、字体、配色、选取阶段/切片、完整性要求 |
| bundle | 压缩方式与完整个体概率保留规则 |

## 必须理解的操作

1. 先核对真实输入路径，不把本机技术示例目录当正式队列。
2. metadata若不需要就设null；配置不存在的文件会明确报错。
3. numeric性别编码须在 `ingest.sex_mapping` 显式添加，例如字符串键 `"1": male`、`"2": female`；程序不猜测数字含义。
4. 输入包模式使用 `cohorts.C3.input_bundle: /实际路径/C3_compact`。导出包已冻结原值和状态，重导入再次核验。
5. 改动参考、模型参数或输入快照时使用新的output_dir；不要删除完成清单以骗过缓存身份检查。
6. 不采用站点效应；site信息即使保留也不作为本项目标准化或SuStaIn协变量。
7. `model.burn_in_fraction`只用于本项目的个体后验与展示抽样，官方CVIC保持官方抽样规则。
8. `report.raster_dpi=600`控制TIFF，`preview_dpi`控制PNG；SVG保留矢量要素。PNG/SVG/TIFF和逐图TSV必须齐全。
9. `assets.template_*`仅用于开发端重建公开模板图谱；正式工作站执行install_display_assets即可。
10. 输入源、输出数据包和展示包均可能含个体ID或概率，始终留在授权工作站，不能放入GitHub。

完整字段解释以analysis.yaml中文注释和各专题文档为准。代码中的稳定产物文件名是接口约定，不属于输入路径配置。

## 已确认的38维与迁移配置

`feature_selection.exclude`列出`feature_id`、`criterion`和`reason`。当前仅排除`wmh_basal_ganglia_layer4_ml`，依据为`LS_primary_all_zero`。程序先核验LS主年龄范围内的原始值确实全零，再冻结38维字典、理由和LS证据。不能仅改expected_model_features来绕过检查。

`feature_selection.merge`完整列出来源特征、目标特征、名称、理由、映射版本和`operation: sum_raw_ml`。最终目标`gm_striatopallidal_basal_forebrain_ml`来自`gm_striatopallidal_ml + gm_basal_forebrain_ml`。先对原始mL求和，再log并拟合LS；不相加或平均旧z。丘脑保持独立。原始GM20编号在`source_atlas_indices`保留，GM19模型编号在`atlas_index`保留。

`config/formal38_migrated.example.yaml`继承工作站`local.yaml`，明确上述排除项、迁移后C3路径和新输出目录。`cohorts.C3.relocate_feature_pointers: true`是显式启用的迁移模式：依据原指针中唯一的本例`sub-ID`后缀，在新的derivatives根下读取对应JSON。禁止`..`、跨受试者、非JSON及符号链接越界；仍核对JSON内participant_id。该模式不修改状态文件。未启用时继续使用原有`source_root`及`path_prefix_map`逻辑。

`scripts/diagnose_reference.py --config config/local.yaml --inputs-only`读取已保存的导入审计，显示三队列名单数、纳入数、排除原因和隐藏ID的错误示例；不修改数据。

# 配置使用说明

## 文件与覆盖顺序

`config/analysis.yaml` 是完整正式参数文件，`config/local.yaml` 是工作站覆盖文件。`extends: analysis.yaml` 做递归字典合并；列表整体替换。相对路径统一以配置的 `project_dir` 为根，不跟随终端当前目录。`config/local*.yaml/json` 不提交Git。

`config/install.json` 控制Python、虚拟环境、锁文件与pip入口；`config/synthetic.yaml` 控制独立技术验证。官方源代码身份由 `vendor/manifest.json` 锁定，wheel和模块源码校验不匹配时必须停止，不静默升级。

## 主要分区

| 配置区 | 控制内容 |
|---|---|
| project_dir / output_dir | 工程根和本次正式结果目录 |
| paths.mapping | 纯GM20固定映射表 |
| cohorts | 每队列derivatives、participants、可选metadata或input_bundle |
| ingest | 自动阶段要求、年龄合法范围、性别映射、ICV一致性 |
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

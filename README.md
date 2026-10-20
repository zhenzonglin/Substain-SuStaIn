# LS–C3–GB WMH/GM SuStaIn

从已经完成的WMH和GM分割结果，建立LS健康参考、C3发现模型和GB冻结验证。主模型使用50–75岁参与者、WMH20＋纯GM20，自动质控通过即可入模，不进行站点校正。所有科研参数和输入路径均在 `config/`；模块按功能分区并有中文注释。

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

默认主规格：40维、z阈值1/2/3、K=1–6、C3十折、25起点、CV每模型10万次MCMC、主模型100万次、两个独立稳定性种子各10万次。完整官方后验可能较大，运行前检查剩余磁盘与配置的内存限额。

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

## 无需搬运原始MRI

在能访问源队列的工作站导出：

```bash
bash scripts/run.sh export-inputs --config config/local.yaml --cohort C3 --destination /可写目录/C3_compact
```

只转移该小型包，在目标配置中设置 `cohorts.C3.input_bundle`。包包含名册、原始特征JSON、GM119/GM20、WMH20、真实ICV、QC与排除审计及哈希；不含MRI或深度学习权重。LS、GB使用相同入口。患者数据包不能上传GitHub。

## 正式结果保留

- `inputs/`：完整导入清单、合格特征、排除原因和来源快照。
- `reference/`：LS参考参数、OOF诊断、冻结变换后的全龄/主分析矩阵。
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

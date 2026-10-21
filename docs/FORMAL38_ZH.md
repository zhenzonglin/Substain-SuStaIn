# 最终WMH19＋GM19工作站部署

本规格为38特征、114事件；仍使用原K=1–6、10折、25起点和正式MCMC次数。它不启动上游分割。

## 固定分区

| 模型区域 | 原始来源 |
|---|---|
| WMH19 | 原WMH20去掉第4区；保留来源编号1–3、5–20 |
| GM1–16 | 原GM1–16，包含独立的丘脑GM16 |
| GM17 | 原GM17纹状体–苍白球＋原GM19基底前脑 |
| GM18 | 原GM18海马–杏仁核 |
| GM19 | 原GM20小脑皮层–蚓部 |

GM合并为原始mL求和，随后log并独立拟合LS参考。完整原始40维保留在inputs，原始GM119仍保留在JSON快照。脑干35及ventralDC61/62不加入。完整模型字典与119标签映射冻结在reference。每次仍要求LS主年龄范围内WMH4确实全零。

## 更新及导入

适用于已经完成安装并有config/local.yaml的工作站。下面先fetch取得升级脚本，因而即使手工修改过analysis.yaml、普通pull被阻止，也可执行。脚本完整备份原文件，把更新前实际生效的路径和参数保存到config/local.preserved_时间戳.yaml，再恢复受版本控制的analysis.yaml并快进更新。最终local.38.yaml继承该保存配置，覆盖最终分区、C3迁移路径与新结果目录；原local.yaml不改。已有local.38.yaml也会先备份。

```bash
cd /data02/linzhenzong/Substain-SuStaIn &&
git fetch origin &&
mkdir -p .cache &&
git show origin/main:scripts/upgrade_formal38.py > .cache/upgrade_formal38.py &&
.venv/bin/python .cache/upgrade_formal38.py --config config/local.yaml &&
bash scripts/run.sh doctor --config config/local.38.yaml &&
bash scripts/run.sh ingest --config config/local.38.yaml &&
.venv/bin/python scripts/diagnose_reference.py --config config/local.38.yaml --inputs-only
```

原始备份与收据位于.cache/config_upgrades/时间戳/。保留config/local.preserved_*.yaml，它是新local.38.yaml的有效继承源。若存在已暂存的analysis.yaml修改，脚本会停止以保留暂存决定；不会reset其他文件。快进失败且HEAD未改变时会恢复原analysis.yaml。

C3配置：

```yaml
cohorts:
  C3:
    derivatives: /mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_C3/derivatives/substain_features
    participants: /mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_C3/participants.tsv
    source_root: /mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_C3
    metadata: null
    input_bundle: null
    relocate_feature_pointers: true
```

导入读取每例状态JSON中记录的真实特征文件相对路径，替换旧工作站根目录，且只允许在本例sub-ID目录内解析；不会修改源JSON或猜测文件名。旧路径即使仍存在，也使用新目录。原阶段失败、实际缺失、权限不足和身份不匹配继续记录排除，不补零。任一队列主年龄范围内为0会阻止reference。

## 正式运行及监控

```bash
mkdir -p logs
nohup bash scripts/run.sh all --config config/local.38.yaml > logs/formal38.log 2>&1 < /dev/null &
```

另一终端查看阶段状态及CPU/内存：

```bash
cd /data02/linzhenzong/Substain-SuStaIn
.venv/bin/python scripts/monitor.py --config config/local.38.yaml
```

all会从ingest运行到report；已有且身份一致的阶段可复用。不要在已有活跃任务时重复启动。已中断的MCMC重新计算，不支持链内续跑。旧维度或旧合并方案的缓存不会被复用。

## 当前4并行CV切换为10并行

```bash
cd /data02/linzhenzong/Substain-SuStaIn &&
git pull --ff-only &&
.venv/bin/python scripts/switch_parallel.py --config config/local.38.yaml --apply
```

本命令自己识别、停止当前配置的CV进程，并后台续跑；不要另开`all`任务。所有切换参数位于`config/parallel10.yaml`：10个外层CV任务，80%可用内存预算，保留2倍内存估算安全系数。预检不足10并行、源码/数据/科学参数改变或已进入fit时会拒绝切换。去掉`--apply`可只预检。

原配置及所有被更新的缓存身份保存在`.cache/parallel_switches/时间戳/`，大型模型文件不会复制或修改。已完成的ingest、reference和完整模型/折继续使用，未保存的MCMC链重算。后台按crossvalidate→fit→validate→report→bundle→verify-bundle执行，任一阶段失败即停止。主模型及稳定性种子仍串行。

切换后重新打开监控，检查“CV并行：配置上限10｜最近启动实际分配10”，并查看实际进程CPU和内存。完成的折会退出计算，因此活跃任务数可以少于10。

```bash
.venv/bin/python scripts/monitor.py --config config/local.38.yaml
```

## 报告完成后保存展示包

```bash
bash scripts/run.sh bundle --config config/local.38.yaml
bash scripts/run.sh verify-bundle --bundle runs/formal_LS_C3_GB_38_v1/bundles/sustain-display-bundle.tar.gz
```

报告在runs/formal_LS_C3_GB_38_v1/report/index.html。完整分析目录留在工作站，不能只保留展示包而删除完整链。开发端合成验证不等于工作站真实数据导入或正式模型完成。

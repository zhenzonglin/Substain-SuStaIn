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

适用于已经完成安装并有config/local.yaml的工作站。新配置继承现有LS/GB设置，单独覆盖本次修改的C3路径、特征合同与结果目录。下面的cp不覆盖已有local.38.yaml；若已经有此文件，应核对其合并来源必须为GM17+GM19。

```bash
cd /data02/linzhenzong/Substain-SuStaIn
git pull --ff-only
cp -n config/formal38_migrated.example.yaml config/local.38.yaml
bash scripts/run.sh doctor --config config/local.38.yaml
bash scripts/run.sh ingest --config config/local.38.yaml
.venv/bin/python scripts/diagnose_reference.py --config config/local.38.yaml --inputs-only
```

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

## 报告完成后保存展示包

```bash
bash scripts/run.sh bundle --config config/local.38.yaml
bash scripts/run.sh verify-bundle --bundle runs/formal_LS_C3_GB_38_v1/bundles/sustain-display-bundle.tar.gz
```

报告在runs/formal_LS_C3_GB_38_v1/report/index.html。完整分析目录留在工作站，不能只保留展示包而删除完整链。开发端合成验证不等于工作站真实数据导入或正式模型完成。

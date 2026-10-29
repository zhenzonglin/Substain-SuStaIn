# 30物理核上的60个逻辑CPU：容量检查与可选切换

现有30进程模式按一个物理核分配一个单线程EM起点。`logical_cpu_limit: 60`限制的是CPU亲和性范围，不会自动产生60个计算进程。`htop`的running是整台机器当时可运行/正在运行的任务数，不能直接当作本项目的完成进度或专属worker数。

新增`config/startpoints60.yaml`使用`worker_unit: logical_cpu`、每折6起点，最多10折同时执行。30物理核和60逻辑CPU的总范围不变。每个起点绑定一个不同的逻辑CPU，先分配不同物理核的第一个线程，再使用SMT兄弟；BLAS保持1。25个总起点不变，主拟合仍最多25个起点，MCMC仍每折一条顺序链。因此60是EM调度上限，不能保证running一直为60。

SMT的两个逻辑CPU共享物理核的计算资源，60进程不等于60物理核，也不保证相对30进程提速。性能应比较同一组计算任务的墙钟时间，不能根据CPU条或running数判断。[Intel关于SMT资源共享的说明](https://www.intel.com/content/www/us/en/support/articles/000098959/processors/intel-xeon-processors.html)。

## 先只读检查，不更新正在运行的源码

无需sudo。以下命令只fetch Git对象、从新提交取出独立检查脚本和配置到`.cache`，不会pull或切换当前代码，也不会停止分析。

```bash
cd /data02/linzhenzong/Substain-SuStaIn &&
git fetch origin &&
mkdir -p .cache/smt60_precheck &&
git show origin/main:scripts/check_smt_capacity.py > .cache/smt60_precheck/check_smt_capacity.py &&
git show origin/main:config/startpoints60.yaml > .cache/smt60_precheck/startpoints60.yaml &&
.venv/bin/python .cache/smt60_precheck/check_smt_capacity.py \
    --config config/local.38.yaml \
    --settings .cache/smt60_precheck/startpoints60.yaml
```

看`full_request_allowed`、`allowed_compute_processes`、`requested_peak_estimate_gib`和`memory_budget_gib`。这是资源估算，尚不是缓存完整性验收或速度测试。检查脚本兼容已发布的`7e38ac1`，配置中的路径和正式参数保持原值。

按6792例、38维、114事件、K上限6、原安全倍数2和可用内存比例0.8估算：每折6起点约20.01 GiB，10折约200.12 GiB；116.21 GiB可用内存对应92.97 GiB预算，只能容纳4折×6=24个计算进程。当前十几GiB的RSS并不是后续候选张量的峰值。此估算是开发端测试结果，不代表重新测量过工作站。

**约125 GiB总内存的机器在这些默认条件下不应强行切换60进程。** 专用配置要求全部请求折数通过资源检查，资源不足在停止之前拒绝。不要为了显示60 running调低内存安全倍数或跳过预检。保留现有30进程模式通常是当前合理选择。

## 资源允许时再迁移

只适用于尚未进入fit的CV任务、已启用完整起点检查点的已发布版本；所有完整K缓存和起点均需校验。

```bash
cd /data02/linzhenzong/Substain-SuStaIn &&
git pull --ff-only &&
.venv/bin/python scripts/switch_startpoints.py \
    --config config/local.38.yaml \
    --settings config/startpoints60.yaml \
    --apply
```

切换先核验身份、资源、原链和起点，并检查全C3的分块似然逐元素一致及合成起点/MCMC串并行一致；然后精准暂停旧协调器和后代进程，再核验一次，备份配置及身份后停止并后台续跑。预检失败不停止旧任务。输入、LS参数、阈值、特征顺序、折分、随机种子和全部EM/MCMC科学规格不变。

原始K模型链的字节和mtime不变。完整旧起点的NPZ与完成回执经校验后复制到新代码身份对应的批次；旧目录原样保留。复制的是轻量单起点结果，不复制大型MCMC链；没有完成回执的起点重新计算，未写盘的MCMC链不能续接。复制或配置写入失败保留旧结果和审计证据，不删除缓存。

备份在`.cache/startpoint_smt_switches/`，日志指针为`logs/formal38_startpoints60.latest_log`。

监控：

```bash
.venv/bin/python scripts/monitor.py --config config/local.38.yaml
```

显示实际折数、每折起点、计算进程上限、`logical_cpu`分配单位以及每个起点的EM进度。进入MCMC或起点批次末尾时running降低是正常调度行为。

已完成迁移后因重启需要续跑时：

```bash
mkdir -p logs &&
resume_log="$PWD/logs/formal38_startpoints60_resume_$(date +%Y%m%d_%H%M%S).log" &&
nohup .venv/bin/python -u scripts/switch_startpoints.py \
    --config config/local.38.yaml --settings config/startpoints60.yaml --resume \
    > "$resume_log" 2>&1 < /dev/null &
```

只在确认旧续跑器已退出且迁移已完成时使用；不要额外运行`all`。完整阶段/起点缓存可复用，文件锁随进程退出释放。

## 软件验证边界

合成测试覆盖60个不重复的CPU槽位、资源足够时放行60/不足时拒绝、原训练函数与7e38ac1一致、三个官方起点入口的串并行等价、完整/部分旧起点恢复、损坏拒绝、完整K原件不变以及继续CV/fit/GB验证。测试不代替工作站实际容量检查、吞吐测试或正式队列结果。

开发端211项测试已分批通过，启用官方集成和38维集成；Ruff、compileall、pip check和无网络wheel构建通过。只读检查脚本还单独在真实7e38ac1源码与合成配置下验证通过，配置原字节保持不变。工作站本轮没有被连接、停止或迁移。

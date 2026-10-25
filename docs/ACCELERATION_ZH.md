# 38维正式模型：分块计算、EM进度与安全切换

此次只改变似然的受试者分批计算和进度记录。LS参考、WMH19＋GM19、114事件、年龄范围、十折、K=1–6、25起点、全部随机种子及正式MCMC次数保持原值。官方pySuStaIn固定提交和安装文件不变，EM/MCMC仍调用官方函数。

## 从旧CV任务切换

适用于目前仍处于crossvalidate的任务；若已进入fit或后续阶段，脚本在停止前拒绝执行并保留现有任务。不要直接修改旧缓存的manifest，也不要同时开第二个all任务。

```bash
cd /data02/linzhenzong/Substain-SuStaIn &&
git pull --ff-only &&
.venv/bin/python scripts/switch_acceleration.py --config config/local.38.yaml --apply
```

无需sudo。默认先核对来源代码、全部科学参数、环境依赖、LS参考、C3输入、折分与种子，并在工作站完整C3矩阵上比较3组有效序列的原始和分块似然。只有有限且逐元素完全一致才继续。当前workers、内存预算与BLAS线程也不在本次迁移中改动。

然后识别同项目、同输出、同用户和解释器的旧CV进程以及后台续跑器，先暂停并再次核查阶段，再保存备份和缓存清单；预检失败会恢复被暂停的进程。验证通过后才终止这一组进程、加锁迁移配置/缓存身份并后台续跑。不会使用截图里的旧PID，不会执行全局pkill。

不加--apply只预检并保存审计，不停止或迁移任务。

## 已有结果如何保留

- 输入、LS参考和完整模型pickle保留原地，不重做ingest/reference，不复制大模型或MRI。
- 每个已有pickle记录哈希、大小、mtime及完整性检查结果。完整缓存继续复用；不完整缓存由已有恢复流程保留为.incomplete文件后重算。
- 本机配置、旧manifest、阶段记录和允许来源的原代码备份到.cache/acceleration_switches/时间戳/；receipt.json列出迁移前后身份及文件。
- 中断的EM起点和未保存的MCMC链不能从内存恢复，必须重跑。本次没有宣称起点级或链内续跑。
- 后台依次运行crossvalidate、fit、validate、report、bundle、verify-bundle；任一阶段失败即停止。配置和缓存已切换且仍有活任务时，再执行同一切换命令不会重复停止。

全部迁移参数位于config/acceleration_switch.yaml。脚本只放行经过测试的旧源码哈希，科学参数或其他来源漂移会拒绝。写入失败时已改文件回退并保留备份；修复具体I/O问题后可重新执行切换命令，不要清空结果目录。

## 查看真实内部进度

```bash
cd /data02/linzhenzong/Substain-SuStaIn
.venv/bin/python scripts/monitor.py --config config/local.38.yaml
```

每折显示当前K、优化批次、起点/25、EM轮、事件/114、候选位置、已完成似然调用及最近logL；MCMC分别显示调优轮和当前迭代评估位置。主模型及稳定性种子使用同一监控方式。

优化批次区分不同层次模型/分裂候选内重新开始的起点搜索，起点号不是整个CV唯一的累计编号。EM轮属于当前EM拟合，分裂初始化时同一起点可包含多个EM拟合。数值计数增长说明对应函数完成了工作；心跳时间更新只说明进度线程存活。MCMC当前迭代仍可能在评估子类型，不能当成链已经落盘。

进度快照位于每个模型目录的progress.json，起点、EM轮、MCMC开始/结束记录追加到progress_events.jsonl；旧事件不覆盖。记录包含PID和创建时间，监控将旧PID标为历史进度，心跳过期标为延迟，不自动认定失败。参数位于分析配置的acceleration区：backend=row_blocks、block_rows=32、heartbeat_seconds=5。

## 数值与性能边界

分块只沿受试者轴切分，保留官方轨迹构造、每例运算及特征求和顺序；不使用改写求和顺序的矩阵乘法近似。源码提取前严格验证固定上游文件哈希。分块和进度模块的源码哈希写入新模型身份。

开发端同规模合成数据（6112人、38特征、114事件、BLAS单线程）三次计时中位数：原始核函数约0.307秒，32人分块约0.070秒，约4.4倍。此数值不是工作站实测速率或整套CV加速倍数。数值一致性检查也不能替代模型收敛和稳定性检验。

独立性能诊断入口为scripts/audit_cv_performance.py --config config/local.38.yaml，参数在config/cv_performance.yaml。它仅在独立实例中测量原始/分块似然和单轮EM；旧probe_cv.py仍使用原始核函数。

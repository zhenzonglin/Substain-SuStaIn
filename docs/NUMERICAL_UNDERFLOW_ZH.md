# 留出似然为零：含义、数值修正及解释边界

某折K=1缓存可能出现“数组形状正确、负值/NaN/Inf均为0，但部分平均似然为0”。对应比例属于该留出折，不能直接作为整个C3队列的比例。似然为0也不等于WMH体积为零或亚型概率为零。

## 这些数值表示什么

每人的整组38维z值与某个阶段/亚型的高斯轨迹比较，得到联合密度；再对阶段、亚型和固定官方抽取的MCMC样本求和或平均。保存字段`mean_likelihood_subj_test`是留出样本的平均模型似然，不是个体疾病诊断概率。高维联合密度通常很小，较小的绝对值本身不能判定模型无效。

float64最小正subnormal约为4.94×10^-324。过小的指数结果可能舍入为0，例如exp(-1000)在线性float64表示中为0，而对应的对数值-1000仍能保存。多维密度相乘、阶段或亚型加权、平均均可能暴露这种数值限制。

可能的诱因包括：某些大正z或大负z、LS残差尺度小、WMH零膨胀或重尾、某个模型无法拟合的特征组合。输入单位、ICV或特征映射错误也应核查；仅凭零值数量不能排除这些问题。Z_max=5约束模型轨迹，不是把观测z自动截到5。离所有轨迹很远的观测，在不同K下都可能产生极小似然。

## 后果

固定官方版本的CVIC直接计算-2×sum(log(平均似然))，其中一个平均似然为0就会得到正无穷；任意加小数会改变尾部样本对CVIC的贡献。因此不能直接补成10^-250，也不能把这些人删去。

官方EM/MCMC内部的训练目标包含log(个体似然+10^-250)。似然远低于这一常数时，其训练目标贡献接近固定值。稳定评估能恢复冻结模型的可比较分数和有效概率归一化，但不会追溯改变训练时的优化或采样。若很多训练样本处于这一范围，需要结合输入质量、LS校准和模型假设判断是否进一步修订训练实现；不能仅凭“程序完成”宣布模型可信。

本项目统计保存ML-EM序列下的训练下限影响人数。它只是一份冻结解的诊断，不代表所有EM/MCMC迭代，也不表示已经重新训练。各折训练集互有重叠，不能把这些训练人数简单相加作为队列人数。

## 本次修正

1. 保留原pickle及其中的零值，不改体积、z、特征、阈值、折分、种子或EM/MCMC。
2. 沿用已核验官方的轨迹构造和高斯算式，在取exp之前保留对数值。使用SciPy的logsumexp完成阶段、亚型及MCMC均值运算；不改变CV的抽样索引或引入展示用burn-in。
3. 预检对原始零值对应的人逐例重算冻结链：重新得到的线性结果必须与缓存相符，稳定对数结果必须有限。任意被错误改为0的缓存不会被自动接受。
4. CVIC使用同一数学公式的稳定表示；原始CVIC若为无穷则记录`official_cvic_status=infinite_due_to_zero`，不以有限数字冒充。原线性均值、原官方带下限的平均log似然、稳定分数和零值人数分别留存。
5. 官方分型/分期API只在评估期间使用稳定比例，返回后立即恢复原训练似然函数。`official_*`保留可计算的原始输出；原始全零导致不能归一化时保留NaN并标记`official_raw_available=false`。subnormal范围仍可计算的原始输出保留，但标记`official_raw_valid=false`。稳定官方定义的输出另存`official_stable_*`，比例加权的联合后验另存`posterior_*`。
6. 报告展示数值处理方式、留出零值人数和保存ML-EM解的训练下限统计，不把数值修复当成参考校准、收敛或外部验证的证据。

稳定评估与普通范围内的原始实现按配置容差核对，不宣称对数运算与线性运算逐位相同。原始训练及起点调度的等价检验仍保持原有要求。

## 工作站命令

先只读检查诱因和数值复核结果：

```bash
cd /data02/linzhenzong/Substain-SuStaIn &&
git pull --ff-only &&
.venv/bin/python scripts/diagnose_likelihood.py --config config/local.38.yaml
```

诊断打印每折K=1的零值数、有限对数似然范围、最大绝对z、主要涉及特征及其LS残差SD，以及保存ML-EM解下的训练下限人数。只写`.cache/heldout_diagnostics/<时间戳>/diagnosis.json`，不停止旧任务。参数在`config/likelihood_diagnostic.yaml`，数值参数在分析配置和切换配置的`numerical_evaluation`区。

保留K=1并立即切换：

```bash
.venv/bin/python scripts/switch_startpoints.py --config config/local.38.yaml --apply
```

切换本身也执行上述零值复核，因此只读诊断不是必须重复运行的安装步骤。已有完整更高K也保留，不等待K=2。原始缓存字节和mtime不改，新的评估记录独立保存。若出现形状、负值、非有限值或原始线性复算不一致，会在停止旧任务前拒绝切换。

正式输出包括各折`heldout_evaluation/k*_audit.json`和`k*_subjects.tsv`，以及汇总的`model/cv/heldout_subjects.tsv`、`fold_loglikelihood.tsv`、`cvic.tsv`和`numerical_evaluation.json`。展示包携带汇总数值、数值诊断及重画代码；全链仍在分析档案。

## 实现依据

开发端已通过196项全套回归测试；随后对极大负对数下的归一化边界补强，并通过28项相关复查。测试覆盖原线性结果复现、对数域结果、官方CV抽样索引、亚型权重、完整缓存原字节/mtime保留、原训练函数恢复、GB推断与只读诊断。真实工作站的零值诱因和训练影响须以现场诊断记录为准。

- [固定UCL官方源码](https://github.com/ucl-pond/pySuStaIn/blob/708fa22d89c9692dbeafcc2c31a1c8460ced5640/pySuStaIn/AbstractSustain.py)：CV抽样、CVIC公式、训练似然保护常数和分型API。
- [SciPy logsumexp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html)：求指数和的对数的稳定实现。
- [NumPy浮点限制](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)：normal/subnormal范围与低精度区域。

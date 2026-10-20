# 验证记录 — 2026-10-18

## 已完成的软件验收

- 完整测试套件：49项通过，包含真实官方SuStaIn的EM/MCMC、两折外层并行和断点恢复；不是用模拟实现代替官方算法。
- 增加展示进程断网限制后，相关10项报告/打包测试再次全部通过。
- Ruff、Python语法编译、sdist/wheel构建通过；pip依赖一致性检查通过。
- 独立Python3.10新环境安装通过。Python网络白名单记录仅出现pypi.org和files.pythonhosted.org；三个GitHub依赖从仓库内固定wheel安装。
- 官方源码与固定commit的关键模块SHA256一致。
- 已验证缺少已有WMH图谱时，可从固定GitHub资源取得小图谱并通过SHA256校验。
- 指定分割父项目原有18项已修改文件和2项未跟踪文件保持原状态；只新增sustain_analysis目录。没有重跑任何患者分割。

## 一体合成集成验收

真实执行了：合成体积→statsmodels/Patsy LS参考→LS五折留出诊断→C3官方SuStaIn→冻结GB→全龄外推→报告→展示包重画。

此测试使用3个特征、100名支持年龄内LS、40名C3、12名GB；C3和GB各另有2名范围外测试样本。测试参数独立保存于config/synthetic.yaml。每个官方模型仍保留3×10000次提议调优；短MCMC只用于接口验收，不能解释为充分混合或正式研究结果。正式输入契约保持40维。

本地输出：runs/synthetic_validation。

- 17组图：17 PNG、17 SVG、17 TIFF。
- 17张对应数值表。
- 全部TIFF独立核验为RGB、600×600 DPI。
- C3/GB × primary/all四套完整个体后验齐全，联合分布与边际一致。
- 真实公开模板空间图已人工视觉检查；每张合成结果图标明SYNTHETIC DATA。

## 展示包验收

包大小9,471,075字节，66个保留文件。

SHA256：
`fdf424a2e0711039c919e70db103536750759348c003e613d82417577adf4617`

验证在独立临时目录完成，使用包内代码、数据与图谱，从没有成品图片的状态重画17组图，核验17张数值表及34张PNG/TIFF的像素或尺寸/DPI。

重画子进程禁止Python socket DNS/connect操作，四项受控网络调用均确认被拒绝；没有修改系统网络设置。重画不需要原MRI或原项目目录。机器可读结果见verification_summary.json；原始收据位于本地展示包旁。

## 尚未执行的正式研究步骤

本机无法访问三队列工作站结果目录，因此没有核实正式样本数，也没有拟合正式LS–C3–GB模型。工作站安装后须按README核验真实路径并执行正式流程。软件/合成验证不能替代真实队列数据质控、参考分布诊断、模型稳定性及外部验证结论。

## 2026-10-20 安装源继承修复

工作站报告了访问pypi.org时的Network is unreachable。原安装器设置PIP_CONFIG_FILE=/dev/null并显式指定pypi.org，会绕过既有工作站pip配置。

- 默认pip_index_url改为null，保留已有源、代理、证书及PIP_CONFIG_FILE。显式覆盖和本地wheel目录分别使用--index-url、--wheelhouse。
- 9项安装专项测试通过；系统Python3.8语法编译及Ruff通过。
- 全新Python3.10环境真实安装通过：以系统Python3.8启动安装器，禁止外部PyPI域名，只沿用PIP_CONFIG_FILE配置的本地测试镜像。共服务90次请求、45个普通依赖wheel，官方工具导入与pip check通过。
- 依赖版本锁、模型算法、患者数据和已有分割环境未变更；工作站本身的镜像连通性仍需现场运行确认。

# 第三方来源与分发边界

- pySuStaIn：UCL POND，提交708fa22d89c9692dbeafcc2c31a1c8460ced5640。
- awkde：noxtoby，提交1c31e55fe54c0cad80ab423a9605fc9ddfb2614c。
- kde_ebm：UCL POND，提交3ad8b648a4a2d0a8df0707f382b54c4ebef0805c。
- 上述小型wheel在vendor/wheels；完整版权文本在vendor/licenses，来源与SHA256见vendor/manifest.json。awkde仅验证Python3.10/Linux x86_64/glibc≥2.31；不冒称通用manylinux wheel。
- ch2better公开模板来自现有mricron-data副本，版权说明随resources/bundled/MRICRON_COPYRIGHT.txt提供。
- GM20展示图谱由公开ch2better模板的NiChart分割结果，经本项目纯GM映射生成。它是ROI展示资源，不是新的人群图谱或个体体素统计结果。NiChart声明保留于resources/bundled/NICHART_NOTICE.txt；没有分发分割权重或软件环境。
- Chung WMH20图谱来自JinyongChung/WMH_progression_modeling提交1d9c8456168fa8b3a0190c6e91ca3dfbc6c90068。因该来源未见仓库级许可，本仓库不重新打包分发；安装脚本复用授权工作站现有副本，或从上游固定GitHub链接取得。
- 患者MRI、名册、原始特征、LS参考参数、SuStaIn后验及真实结果均不在GitHub中。

使用前应遵守各来源许可及研究机构的数据管理要求。软件用于研究，不提供临床诊断或治疗建议。

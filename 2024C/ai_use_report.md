AI 使用报告（简要）

范围：本次工作中使用的 AI / 自动化工具与模型

- 代码与推断：使用 `pymc`（PyMC v5）做贝叶斯建模与 ADVI 变分推断；使用 `arviz` 做后验汇总与 HDI 计算。
- Notebook 自动化：通过 Jupyter Notebook (`EDA.ipynb`) 组织 EDA 与模型单元，所有实验可在 notebook 中复现。
- 自动脚本：在工作区生成并运行了多段 Python 单元，包含基线逻辑回归、EDA、短次与长次 ADVI 批量推断。

注意与局限：
- ADVI 为近似推断，可能低估后验不确定性；对关键参数（alpha、rho）建议在 1–2 场使用 MCMC 进一步验证。
- 运行环境在本地 Python 中完成（无远端 GPU/MPI 加速）；缺少 `g++` 可能影响 PyTensor 的速度。

可复现说明：
- 复现步骤：在工作区打开 `EDA.ipynb`，确保安装依赖（见 notebook kernel 已列出主要包），按顺序执行单元。
- 重要脚本与输出路径已经写入 `summary_sheet.md`。
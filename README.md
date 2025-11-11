# xinxifenxi3

该仓库提供一个用于比较不同机器学习模型在归一化前后表现差异的脚本。

## 功能概述

`analysis.py` 会读取用户提供的 CSV 数据集，按照 8:2 划分训练集与测试集，
并在训练集上进行 K 折交叉验证（默认 5 折）。脚本会针对多种模型（包括
Logistic Regression、Random Forest，以及若安装则包含 XGBoost 与 LightGBM）
分别构建“归一化”和“未归一化”两条流水线，比较其得分差异并指出对归一化最敏感的模型。

## 使用方法

```bash
python analysis.py --data <数据集路径> --target <目标列名> \
    --task <classification|regression> --folds <折数> --output <输出CSV路径>
```

- `--data`：CSV 数据集的路径。
- `--target`：目标变量列名。
- `--task`：任务类型，可选 `classification`（默认）或 `regression`。
- `--folds`：交叉验证折数，默认 5。
- `--output`：可选参数，如提供则会把结果表格写入该路径。

执行完成后，终端将打印各模型在归一化前后的交叉验证与测试集得分差异，并指出对归一化最敏感的模型。

## 依赖

- pandas
- numpy
- scikit-learn
- 可选：xgboost、lightgbm（若安装则会自动使用，对应模型在未安装时会自动跳过）

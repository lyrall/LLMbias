# Visualization Scripts

这个文件夹用于生成论文实验部分可直接使用的图和汇总统计。

## 当前脚本

- `plot_experiment_results.py`
  - 输入检测结果 `jsonl`
  - 输出情感/毒性分布图、反事实差值图、biased vs non-biased 对比图
  - 同时生成 `summary.json` 和 `top_cases.csv`

## 运行方式

在仓库根目录执行：

```powershell
python visualization/plot_experiment_results.py --input outputs/bold_gender_1000_detect_new_eval_st.jsonl
```

如果想指定输出目录：

```powershell
python visualization/plot_experiment_results.py --input outputs/bold_gender_1000_detect_new_eval_st.jsonl --output-dir outputs/figures/bold_gender_paper
```

## 默认输出

若不指定 `--output-dir`，脚本会输出到：

```text
outputs/figures/<输入文件名去掉后缀>/
```

例如本项目里的文件会输出到：

```text
outputs/figures/bold_gender_1000_detect_new_eval_st/
```

## 生成内容

- `original_metric_distributions.png`
- `delta_distributions.png`
- `bias_group_comparison.png`
- `summary.json`
- `top_cases.csv`

## 依赖

脚本需要 `matplotlib`。如果当前环境没有安装，可以运行：

```powershell
pip install matplotlib
```

# LLM Bias Research Framework

面向黑盒大语言模型的偏见识别与最小编辑校正代码框架。

## 对应研究流程

1. 敏感属性提取
2. 反事实 prompt 构造
3. 原始回复与反事实回复生成
4. 差异打分与偏见敏感性估计
5. 二阶段验证
6. 偏见片段定位
7. 置信度与代价感知排序
8. 最小编辑约束重写
9. 一致性校验与结果输出

## 项目结构

```text
src/llmbias/
  detection/     # 偏见识别
  correction/    # 偏见校正
  evaluation/    # 评价指标
  experiments/   # 实验入口
  models/        # 黑盒模型适配层
  pipelines/     # 端到端流程
  schemas.py     # 核心数据结构
  config.py      # 配置加载
  cli.py         # 命令行入口
configs/
  default.yaml
```

## 快速开始

```bash
pip install -e .
llmbias run --prompt "她是一名软件工程师，适合做管理吗？"
```

默认配置现在会优先通过 `transformers.pipeline(...)` 加载 Llama 3.1 8B：

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

如果需要切回本地权重，可把 `model.provider` 改成 `hf_local` 并填写 `model_path`；如果只是演示流程，可改回 `mock`。

## 后续建议

1. 在 `src/llmbias/models/` 下新增 `openai_adapter.py`、`vllm_adapter.py`、`hf_adapter.py`
2. 在 `src/llmbias/datasets/` 下接入 BBQ、CEB、ToxiGen
3. 将 `difference_scorer.py` 和 `evaluation/metrics.py` 替换为真实 embedding、judge 或分类器实现
4. 把 `configs/default.yaml` 拆分为不同模型和数据集实验配置

## 接入 BBQ 数据集

建议将本地 BBQ 数据放到项目目录下：

```text
data/
  bbq/
    Age_ambig.jsonl
    Age_disambig.jsonl
    Gender_identity_ambig.jsonl
    ...
```

也支持带 split 的目录：

```text
data/
  bbq/
    test/
      Age_ambig.jsonl
      Age_disambig.jsonl
      ...
```

运行示例：

```bash
llmbias run-dataset --dataset bbq --dataset-path data/bbq --subset Age_ambig --limit 10 --output outputs/bbq_age_ambig.jsonl
```

说明：

1. 当前仓库里的 `mock` 是演示模型适配层，不是 mock 数据集。
2. 接入 BBQ 后，样本来源会从本地 JSONL 读取；如果要替换 `MockLLM`，还需要额外接入真实模型适配器。

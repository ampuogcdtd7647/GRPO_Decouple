## 项目介绍：结构感知的语言模型强化学习微调框架

**GRPO_Decouple** 是一个基于 HuggingFace [TRL](https://github.com/huggingface/trl) 框架的强化学习训练项目，采用改进版的 GRPO（Generalized Reparameterized Policy Optimization）算法，旨在提升语言模型在 **结构化推理任务** 中的输出质量与准确率。

本项目通过设计结构感知的奖励函数，引导模型生成标准化的推理格式，并对输出的逻辑完整性、答案正确性、长度分布等方面进行强化学习优化。

---

### 使用说明：替换 TRL 库并引入自定义 GRPOTrainerModify

本项目对 HuggingFace 官方的 [`trl`](https://github.com/huggingface/trl) 库进行了定制扩展，主要目标是增强对结构化奖励、格式控制、多奖励组合等功能的支持。

#### 第一步：覆盖原始 `trl` 库源码

请将项目中的 [`modify_trl/`](./modify_trl) 文件夹下的内容，**手动替换你当前 Python 环境中的 `trl` 源码目录**，以便正确使用自定义 Trainer 类。

#### 第二步：在 `trl` 库的trainer文件夹中新建GRPOTrainerModify.py文件

GRPOTrainerModify 是本项目自定义的训练器类，继承并扩展了 HuggingFace GRPOTrainer，具备以下增强特性：
- ##### 支持多个自定义奖励函数的组合训练（结构 + 格式 + 正确性 + 长度）
- ##### 支持自定义 tokenizer 注入，便于长度奖励评估
在`train_grpo.py` 中使用方式如下：
```python
from trl.trainer import GRPOTrainerModify

trainer = GRPOTrainerModify(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        length_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
    **{"tokenizer": tokenizer},
)

```
---
### 核心亮点与创新

- **自定义 GRPOTrainerModify**
  - 支持多种奖励函数的并行组合与注入
  - 增强格式校验与输出控制能力
  - 解耦裁剪比率

- **多维奖励函数设计**
  - `correctness_reward_func`：答案正确性评估
  - `int_reward_func`：是否为合法整数
  - `strict/soft_format_reward_func`：格式正则匹配校验
  - `xmlcount_reward_func`：结构标签数量与位置奖励
  - `length_reward_func`：生成长度正态分布控制（μ=200）

---


### 项目适用场景

- 推理任务微调
- 结构化输出对齐任务（如代码生成、XML/JSON 风格生成）
- 多奖励函数融合的 RLHF 微调实验
- 格式敏感的指令微调训练场景


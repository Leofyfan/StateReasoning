# StateReasoning

人类在不同的情绪/状态下（比如兴奋，悲伤，焦虑）对同一个问题的思考/决定的结果往往是不同的。
而且人类在做题目的时候也往往会伴随着情绪（状态的变化），比如在做压轴难题的时候，状态的变化是刚接触题目(不确定，紧张，会反复读题，保证正确理解题目) -> 没有头绪的焦虑（到处想解题思路） ->发现思路的激动-> 逐步解题完成的自信 -> 重新检查，自我怀疑
即整个过程伴随着的状态的动态变化，而且不同的状态对应的特点是不一样的。那么这种动态变化思考的范式能不能迁移到视觉语言模型中去呢？

比如在读题阶段，让模型更专注，反复读题，保证正确理解题目 （）。
没有头绪的思考阶段，也降低置信度，增强模型的探索能力（）。
发现思路的激动，此时应该对应着一个模型置信度骤升的阶段。（自信程度骤升，然后有个稳步下降的过程）
逐步解题完成的自信，应该对应着一个confidence 平稳提高的过程。
重新检查，自我怀疑（开始检查时是不是有个confidence骤降的过程，能帮助更好地纠错？或者说该过程应该对之前的回答进行怀疑，模型对之前的行为保持不确定），

上述的这种的idea 是猜想，可以通过 置信度 来体现这种状态吗？

置信度评估标准
相对置信度分数（更适合量化状态的变化？）和绝对置信度分数（全局的知识掌握？）

Logit-based（基于概率）方法：

Verbalized Confidence（语言化置信度）方法：直接让模型说出“我有多少把握”.
Verbalized 语言化置信度直接提示 LLM 对答案及其置信度分数进行推理。
Verbalized Top-k 提示模型生成 k 个猜测，并附上每一个猜测的概率。随后，概率最高的预测会被选为最终输出。

logit 和 Verbalized Confidence 这种 是一致的吗？感觉会有gap（需要验证一下）

ECE

## MathVista evaluation with VERL

`scripts/mathvista_eval.py` 使用 VERL 的工作流模板驱动 Qwen3-VL-4B-Thinking 在 MathVista 数据集上进行推理，同时跟踪 logit-based 置信度，生成相对置信度曲线并将全过程记录到 Weights & Biases。

### 准备环境
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 确保具备访问 MathVista 数据集与 Qwen3-VL-4B-Thinking 模型的网络权限与凭证。

### 运行示例
```bash
python scripts/mathvista_eval.py \
  --model-id Qwen/Qwen3-VL-4B-Thinking \
  --split testmini \
  --limit 8 \
  --project mathvista-verl \
  --run-name qwen3-vl-thinking-test
```

- `--split` 可设置为 `testmini` 或 `test`。
- `--limit` 控制样本数量，便于快速验证。
- 运行结束后，W&B 中的表格会包含每道题的答案、预测、绝对/相对置信度以及五阶段（read/explore/insight/solve/verify）的置信度轨迹。

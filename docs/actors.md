# Actor 设计

本文介绍当前 Nano Agentic RL 的 Actor 化架构。系统使用 Monarch 作为控制层，将训练、推理、数据、奖励、优势计算和 replay buffer 拆成独立 Actor。每个 Actor 管理自己的状态，controller 只负责调度主流程。

## 总体结构

当前有三个入口：

- `main_sft.py`：SFT 训练和 eval。
- `main_rl.py`：同步 GRPO。
- `main_rl_async.py`：异步 GRPO。

三个入口复用同一套核心 Actor：

```text
Monarch controller
  |
  |-- DatasetActor(train)
  |-- DatasetActor(eval)
  |-- RewardActor
  |-- AdvantageActor          # RL 使用
  |-- ReplayBufferActor       # RL 使用
  |
  |-- TrainActor mesh
  |     `-- VeOmni/FSDP2 ranks
  |
  `-- RolloutActor
        `-- vLLM AsyncLLM
```

SFT 不启动 `AdvantageActor` 和 `ReplayBufferActor` 的训练路径。它可以看作一个没有 rollout/replay/advantage 的简化 RL 流程：数据来自 dataset actor，训练由 train actor 执行，eval 仍由 rollout actor + reward actor 完成。

## DatasetActor

`DatasetActor` 是数据服务。构造参数为：

```python
DatasetActor(config_path, dataset_name="train" | "eval")
```

它读取 `dataloader.train.path` 或 `dataloader.eval.path`，加载预处理后的 jsonl 数据。当前数据要求包含：

- `messages`
- `label`

对 rollout/eval，actor 会移除 assistant message，只保留 system/user message 给 vLLM。`target` 使用 `label` 字段，`prompt` 使用最后一条 user message 的内容。

主要 endpoint：

- `setup()`：加载数据和 tokenizer。
- `next_batch(batch_size)`：给训练 rollout 或 SFT 取下一批样本。
- `all_batches(batch_size)`：给 eval 遍历完整验证集。
- `count_tokens(messages_batch)`：统计 prompt token 数，用于 dynamic sampling 的长度过滤。
- `get_pad_token_id()`：提供 replay buffer padding 所需 pad token。

## TrainActor

`TrainActor` 是 VeOmni 训练侧封装。每个训练 GPU 对应一个 Monarch actor rank，多个 rank 组成 train mesh。

当前 `TrainActor` 不再把 VeOmni dataloader/epoch 当作 RL 的控制语义，而是使用项目内 stepwise trainer：

- `VeOmniGRPOTrainer`
- `VeOmniSFTTrainer`

VeOmni 仍负责模型构建、FSDP2、optimizer、scheduler、gradient checkpointing、checkpoint callback 等训练运行时能力。

主要 endpoint：

- `setup()`：解析配置，初始化 VeOmni runtime。
- `train_grpo_step(batches)`：执行一个 GRPO optimizer step。
- `train_sft_step(sample_batches)`：执行一个 SFT optimizer step。
- `get_weight_transfer_metadata()`：返回要同步给 vLLM 的参数名、shape、dtype。
- `init_weight_transfer(...)`：初始化 trainer 侧 NCCL weight transfer。
- `broadcast_weights()`：所有 FSDP rank all-gather 参数，rank0 发送给 vLLM。

GRPO batch contract 包含：

- `tokens`
- `attention_mask`
- `position_ids`
- `generator_logprobs`
- `loss_mask`
- `advantages`

GRPO loss 使用 token-level normalization，按当前 rank batch 的 active response token 总数归一。

## RolloutActor

`RolloutActor` 是 vLLM 推理侧封装。它在一个 Monarch actor 内启动 vLLM `AsyncLLM`，vLLM 内部可以使用 DP/TP/PP worker。

主要 endpoint：

- `setup()`：读取 `rollout_actor.engine`，启动 vLLM。
- `chat(conversations, sampling_params=None)`：使用 chat template 批量采样。
- `generate(prompts, sampling_params=None)`：文本 prompt 采样。
- `init_weight_transfer(...)`：初始化 vLLM 侧 NCCL receiver。
- `receive_weights(metadata, version=...)`：接收 trainer 广播的新权重。
- `get_status()`：返回 policy version 和 rollout actor 状态。

rollout 和 eval 使用同一个 actor，区别只在 sampling 参数：

- rollout 使用 `rollout_actor.rollout.sampling`
- eval 使用 `rollout_actor.eval.sampling`

为了支持异步权重更新，生成阶段不长期持有 actor lock。权重更新时，actor 会关闭 generation gate，调用 vLLM `pause_generation(mode="abort")` 中断 in-flight request，更新完成后再 `resume_generation()`。

## RewardActor 和 AdvantageActor

`RewardActor` 提供规则奖励。当前主要面向数学答案：

- 从 `<answer>...</answer>` 中抽取预测答案。
- 与 label 做数值匹配。
- 可选 format reward。

`AdvantageActor` 对每个 rollout group 计算组内标准化 advantage：

```text
advantage_i = (reward_i - group_mean) / (group_std + epsilon)
```

低方差 group 可由 dynamic sampling 过滤。全对或全错 group 没有有效策略梯度，通常不会进入训练。

## ReplayBufferActor

`ReplayBufferActor` 保存 rollout 产生的 `RLEpisode`，并构造 VeOmni rank-local batch。

它按 episode 存储，而不是按 group 存储。一个 group 通常有 `sampling.n` 条 episode，但 sample 和 consume 都是 episode-level，所以 buffer size 不保证是 `n` 的倍数。

主要行为：

- 根据 `capacity` 控制最大容量。
- 根据 `max_policy_age` 清理过旧 policy 的样本。
- 根据 `global_batch_size / train_actor.num_gpus` 为每个 rank 构造 batch。
- `consume_samples: true` 时，sample 后删除被选中的 episode。
- 根据 `monarch.sequence.max_prompt_tokens/max_response_tokens` 做 padding/truncation。
- `mask_truncated: true` 时，对 finish reason 为 `length` 的 response 清空 loss mask。

## Controller 编排

controller 负责把 Actor 串成流程：

- 设置每个 ProcMesh 的环境变量和 `CUDA_VISIBLE_DEVICES`。
- 给 train mesh 配置 Torch Elastic 分布式环境。
- 启动 support actors、train actors 和 rollout actor。
- 初始化 NCCL weight transfer。
- 执行 baseline eval。
- 根据入口脚本选择 SFT、同步 RL 或异步 RL 主循环。
- 记录 W&B 和运行时指标。
- 统一清理 Monarch 进程和 actor 资源。

Actor 化让主流程保持清晰：controller 不直接处理模型细节，训练不直接关心 vLLM，vLLM 不直接关心 reward/replay。每个组件只暴露少量 endpoint，通过 Monarch RPC 组合成完整系统。

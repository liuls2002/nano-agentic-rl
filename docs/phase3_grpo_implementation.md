# 阶段三：Monarch + VeOmni + vLLM GRPO 训练原型

## 1. 阶段目标

本阶段在阶段一 Train Actor 和阶段二 Rollout Actor 的基础上，补齐最小 RL 训练闭环：

- Dataset Actor：提供可复现的 GSM8K prompt 数据流。
- Reward Actor：计算答案正确性奖励。
- Advantage Actor：计算组内标准化 GRPO advantage。
- Replay Buffer Actor：保存 rollout episode 并构造 VeOmni rank-local batch。
- GRPO Train Step：使用 rollout logprob 和 advantage 更新训练模型。
- `main_rl.py`：通过 Monarch 编排 rollout、reward、训练和权重同步。
- NCCL 权重直传：训练完成后直接将 VeOmni FSDP2 权重广播给 vLLM DP workers。

本阶段只实现基础文本 GRPO 原型，不包含 agent environment、工具调用、多轮 trajectory、通用 reward model 或上层 Agent Demo。

## 2. 完成文件

主要新增或修改文件如下：

```text
nano-agentic-rl/
├── actor/
│   ├── dataset_actor.py
│   ├── replay_buffer_actor.py
│   ├── reward_advantage_actor.py
│   ├── rollout_actor.py
│   └── train_actor.py
├── config/
│   └── qwen2_5_1_5b_gsm8k_sft.yaml
├── docs/
│   └── phase3_grpo_implementation.md
├── rl/
│   ├── loss.py
│   └── types.py
├── test/
│   ├── test_nccl_weight_sync.py
│   └── test_rl_actors.py
└── main_rl.py
```

## 3. 总体架构

当前使用 Monarch 创建和管理以下进程：

```text
Monarch Controller: main_rl.py
    |
    |-- DatasetActor
    |-- RewardActor
    |-- AdvantageActor
    |-- ReplayBufferActor
    |
    |-- TrainActor mesh
    |     |-- VeOmni/FSDP2 rank 0: GPU 0
    |     `-- VeOmni/FSDP2 rank 1: GPU 1
    |
    `-- RolloutActor
          `-- vLLM AsyncLLM
                |-- DP rank 0: GPU 2
                `-- DP rank 1: GPU 3
```

训练和推理分别占用独立 GPU。Monarch 负责进程生命周期、RPC、错误传播和关闭；VeOmni 负责模型训练；vLLM 负责采样和内部 DP 调度。

## 4. 当前 RL 数据流

每个训练 step 的完整流程为：

1. Replay Buffer 检查是否已有当前 policy version 的完整训练 batch。
2. Dataset Actor 返回一批 prompt。
3. Rollout Actor 使用当前 vLLM policy 采样回答和 token-level logprob。
4. Reward Actor 为每条回答计算 reward。
5. Advantage Actor 在每个 prompt 的回答组内计算标准化 advantage。
6. Episode 写入 Replay Buffer。
7. Replay Buffer 凑齐全局 batch 后，按 VeOmni DP rank 切分。
8. 所有 Train Actor rank 共同执行一个 GRPO optimizer step。
9. 所有 FSDP2 rank 聚合参数，训练 rank 0 通过 NCCL 广播给全部 vLLM workers。
10. vLLM 清理 prefix cache 并递增 policy version，开始下一轮 rollout。

当前配置为：

```yaml
monarch:
  rl:
    rollout_batch_size: 2
    max_rollout_groups_per_step: 16
    max_prompt_tokens: 512
    max_response_tokens: 256
    replay_buffer:
      batch_size_per_rank: 8

veomni:
  data:
    max_seq_len: 768
  train:
    global_batch_size: 16

vllm:
  engine:
    data_parallel_size: 2
  sampling:
    n: 8
    max_tokens: 256
```

因此一次 rollout 获取 2 个 prompt，每个 prompt 生成 8 条回答，共得到 16 条 episode，正好对应一个全局训练 batch。

vLLM 会将 `n=8` 展开为 8 个独立的 `n=1` 子请求，再由内部 DP load balancer 分发给两个 DP ranks。`rollout_batch_size=2` 不代表只有一个 vLLM DP rank 工作。

## 5. Dataset Actor

`DatasetActor` 读取 GSM8K parquet 数据，并提供确定性的 shuffled batch。

主要行为：

- 从 `veomni.data.train_path` 加载 parquet。
- 使用 `data_adapter.prompt_key` 和 `response_key` 读取问题与答案。
- 将 GSM8K 的 `#### final_answer` 转换为最终 target 文本。
- 组合 system prompt 和 user message，形成 vLLM Chat 输入。
- 使用 `seed + epoch` 重新打乱数据顺序。
- 返回稳定的 `epoch-N-row-M` sample ID。

Dataset Actor 同时加载 tokenizer，并将 pad token ID 提供给 Replay Buffer。

## 6. Reward 和 Advantage Actor

### 6.1 Reward Actor

当前 reward 面向 GSM8K，属于简单的规则型二值奖励：

```text
预测的最后一个数字与 target 数字相等 -> correctness = 1
否则                              -> correctness = 0
```

支持从 `<answer>...</answer>` 中提取答案；不存在标签时，从完整回答中提取最后一个数字。当前配置为：

```yaml
reward:
  correctness_weight: 1.0
  format_weight: 0.0
  tolerance: 1.0e-6
```

因此实际 reward 主要是 0 或 1，推理过程、部分正确和回答质量暂时没有额外分数。

### 6.2 Advantage Actor

每个 prompt 的 8 条回答独立构成一个 GRPO group。advantage 按组内 reward 标准化：

```text
advantage_i = (reward_i - group_mean) / (group_std + epsilon)
```

如果 8 条回答全部正确或全部错误，则 `group_std=0`，该组所有 advantage 都为 0，因而没有训练梯度。

`minimum_group_std` 用于判断低方差 group。配置项：

```yaml
drop_low_variance_groups: false
```

当前为 `false`，所以低方差 group 仍会进入训练 batch。改成 `true` 后，控制器会持续采样新 prompt，直到凑齐有效 batch；如果检查了 `max_rollout_groups_per_step` 个 group 后仍不足，则终止当前训练并报告错误。

## 7. Episode 和 Replay Buffer

`RLEpisode` 保存一条 rollout trajectory 所需信息：

- prompt 和 target。
- response 文本及 token IDs。
- rollout policy 产生的逐 token logprob。
- reward、reward breakdown 和 advantage。
- policy version 和 finish reason。

Replay Buffer 负责：

- 按容量保存 episode。
- 根据 `max_policy_age` 清除旧 policy 生成的样本。
- 随机抽取一个全局 batch。
- 将全局 batch 切成一个 rank-local batch 列表。
- 对 prompt 左侧 padding，对 response 右侧 padding。
- 将 prompt 截断到 512 token，将 response 截断到 256 token。
- 对齐 rollout logprob、next-token target 和 loss mask。

当前配置 `max_policy_age: 0`，只允许使用当前 policy version 的 on-policy 样本；`consume_samples: true` 表示抽样后的 episode 会从 buffer 中删除。

当前不会对重复回答去重。重复轨迹仍属于策略采样分布的一部分，会作为独立 episode 进入 buffer。

## 8. GRPO Loss 和训练步骤

`rl/loss.py` 实现 token-level clipped GRPO objective：

```text
ratio = exp(current_logprob - rollout_logprob)
loss  = max(-ratio * advantage, -clipped_ratio * advantage)
```

只在 response token 的 loss mask 上计算损失。Train Actor 将 rank-local batch 按 VeOmni `micro_batch_size` 切成 micro batches，执行：

1. VeOmni/FSDP2 forward。
2. GRPO loss。
3. gradient accumulation。
4. gradient clipping。
5. optimizer step 和 scheduler step。

返回指标包括：

- loss。
- ratio mean。
- approximate KL。
- clip fraction。
- active response tokens。
- gradient norm。
- learning rate。

组内标准化 advantage 的均值接近 0，且刚同步后的 rollout/current policy ratio 接近 1，因此日志中的标量 loss 可能接近 `0.000000`。这不一定代表梯度为 0；应结合 gradient norm 判断。只有 advantage 全为 0 时，该组才确实没有策略梯度。

## 9. NCCL 权重直传

早期流程需要 Train Actor 保存 HuggingFace checkpoint，再由 vLLM 从磁盘加载。本阶段将 RL 主路径改为 vLLM 原生 NCCL weight-transfer API，不再经过 checkpoint。

### 9.1 初始化

Controller 创建一个本机 rendezvous 地址和端口，并发调用：

- Train rank 0：`NCCLWeightTransferEngine.trainer_init`，作为 transfer rank 0。
- vLLM workers：`init_weight_transfer_engine`，从 transfer rank 1 开始加入。

当前 vLLM `DP=2, TP=1, PP=1`，因此 transfer world size 为：

```text
1 个 trainer sender + 2 个 vLLM workers = 3
```

### 9.2 FSDP2 权重聚合与广播

VeOmni 使用 FSDP2，模型参数为 DTensor shard。每次同步时：

1. 所有训练 rank 以相同顺序遍历 `named_parameters()`。
2. 所有 rank 调用 `DTensor.full_tensor()` 参与 collective。
3. 训练 rank 0 将浮点参数转换为 vLLM 配置的 bf16。
4. rank 0 使用 packed NCCL broadcast 发送给所有 vLLM workers。

metadata 和参数迭代使用相同顺序，包含 parameter name、dtype 和 global shape。

### 9.3 vLLM 接收协议

Rollout Actor 在互斥锁内执行：

1. `pause_generation(mode="abort")`。
2. `start_weight_update(is_checkpoint_format=True)`。
3. `update_weights(...)` 接收 NCCL 参数。
4. `finish_weight_update()`。
5. reset prefix cache。
6. 更新 policy version。
7. resume generation。

Controller 必须并发启动 trainer broadcast 和 vLLM receive；NCCL sender/receiver 任一侧单独调用都会等待另一侧加入。

当前配置：

```yaml
weight_sync:
  backend: nccl
  packed: true
  packed_buffer_size_bytes: 268435456
  packed_num_buffers: 2

vllm:
  engine:
    weight_transfer_config:
      backend: nccl
```

本机 CUDA 13.0 环境没有 IB，因此 Monarch worker 继承：

```yaml
NCCL_IB_DISABLE: "1"
```

### 9.4 RotaryEmbedding warning

权重更新时 vLLM 可能打印：

```text
RotaryEmbedding: Failed to load weights
```

Qwen2 的 RotaryEmbedding 保存的是根据模型配置动态构造的 cos/sin cache，在 vLLM 中属于 `persistent=False` 的非持久化 buffer，不在 HuggingFace checkpoint 和 NCCL parameter metadata 中。vLLM 会保留或重新生成该 cache，因此该 warning 不表示训练参数同步失败。

如果 Linear、Embedding 等普通可训练层出现相同 warning，才需要检查 parameter name 和 metadata。

## 10. 配置校验

`main_rl.py` 在启动 GPU actor 前校验：

- 训练 GPU 与 rollout GPU 不得重叠。
- `max_prompt_tokens + max_response_tokens` 不得超过 VeOmni `max_seq_len`。
- 同样不得超过 vLLM `max_model_len`。
- `max_response_tokens` 必须覆盖 `sampling.max_tokens`。
- GRPO 要求 `sampling.n >= 2`。
- `batch_size_per_rank * train DP` 必须等于 VeOmni `global_batch_size`。
- `global_batch_size` 必须可被 `sampling.n` 整除。
- vLLM DP、TP、PP 所需 GPU 数必须等于 `monarch.rollout.gpu_ids` 数量。
- RL 主路径要求 NCCL weight-transfer 配置有效。

当前 token 配置为：

```text
最大输入       = 512
最大输出       = 256
训练总序列长度 = 768
vLLM 上下文长度 = 768
每个 prompt 采样 = 8
```

## 11. 错误处理和关闭

`main_rl.py` 延续阶段一的 Monarch 错误管理方式：

- 注册 Monarch `unhandled_fault_hook`，任一 mesh 失败会通知 Controller。
- 捕获 SIGINT 和 SIGTERM，第一次 Ctrl+C 触发协作式关闭。
- 正在等待 Actor RPC 时同时监听 shutdown event。
- 先调用 Rollout/Train Actor 的 `close`，再停止各 ProcMesh。
- 最后调用 `shutdown_context()`。
- 第二次 Ctrl+C 可强制退出。

Train Actor 关闭 NCCL transfer communicator 和 VeOmni process group；Rollout Actor 关闭 AsyncLLM、DP Coordinator、EngineCore、workers 和 tokenizer background tasks。

## 12. 测试

### 12.1 CPU 组件和协议测试

运行：

```bash
cd /ssd/liuls/projects/hw/nano-agentic-rl
HF_HOME=/tmp/nano-agentic-rl-hf-cache \
  /ssd/liuls/miniconda3/envs/rl/bin/python test/test_rl_actors.py --local
```

覆盖内容：

- GRPO loss 可前向和反向。
- Dataset、Reward、Advantage 和 Replay Buffer endpoint。
- 16 条 episode 的 rank-local batch 构造。
- 512+256 token 对齐和 loss mask。
- vLLM NCCL 接收协议调用顺序。
- policy version 更新。

### 12.2 NCCL GPU smoke test

运行：

```bash
cd /ssd/liuls/projects/hw/nano-agentic-rl
/ssd/liuls/miniconda3/envs/rl/bin/python test/test_nccl_weight_sync.py
```

测试会启动 2 个 VeOmni/FSDP2 ranks 和 vLLM DP=2，并执行一次完整权重同步。实际测试结果：

```text
NCCL weight sync passed: tensors=338 elapsed=1.47s policy=v1
```

### 12.3 静态检查

以下文件通过 `py_compile`：

```text
actor/train_actor.py
actor/rollout_actor.py
main_rl.py
test/test_rl_actors.py
test/test_nccl_weight_sync.py
```

当前环境没有安装 `ruff`，因此未执行 ruff lint。

## 13. 运行 RL

```bash
conda activate rl
cd /ssd/liuls/projects/hw/nano-agentic-rl
python main_rl.py config/qwen2_5_1_5b_gsm8k_sft.yaml
```

正常日志会依次包含：

- Dataset、Reward、Advantage 和 Replay Buffer 初始化。
- VeOmni FSDP2 ranks 初始化。
- vLLM DP workers 初始化。
- NCCL policy transfer group 初始化。
- 每个 rollout group 的 mean reward、std 和 buffer size。
- 每个 GRPO step 的 loss、policy version 和 NCCL sync 时间。

## 14. 当前限制

1. Reward 仅适用于 GSM8K 数值答案，并且主要是 0/1 二值信号。
2. 低方差 group 是否丢弃依赖配置；全部正确或全部错误时 advantage 为 0。
3. 回答文本和 token 序列不会去重。
4. Replay Buffer 是单进程内存实现，不支持持久化和故障恢复。
5. 当前是严格同步的 rollout -> train -> weight sync 流程，没有异步 rollout 或流水线并行。
6. 当前只实现 on-policy、单步 GRPO，没有 reference model KL penalty、value model 或 PPO epoch。
7. Controller 日志当前主要展示 rank 0 的训练结果，尚未实现跨训练 rank 的指标聚合。
8. Reward extraction 使用最后一个数字，回答末尾出现无关数字时可能误判。
9. NCCL rendezvous 当前使用单机 `127.0.0.1`，多机部署需要改成各节点可访问的地址。
10. Agent environment、工具执行、多轮轨迹和 SQL/Claude Code Demo 留待后续阶段。

## 15. 阶段结论

阶段三已经形成可运行的最小闭环：

```text
GSM8K prompt
  -> vLLM DP rollout
  -> rule reward
  -> group-relative advantage
  -> replay buffer
  -> VeOmni FSDP2 GRPO step
  -> NCCL direct weight transfer
  -> next policy rollout
```

该实现验证了 Monarch 可以同时管理训练、推理和 RL 配套 actors，VeOmni 可以承担多卡 Train Actor，vLLM 可以承担内部 DP Rollout Actor，并通过 NCCL 完成训练权重到推理权重的低延迟同步。

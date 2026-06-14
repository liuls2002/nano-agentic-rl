# 阶段二：基于 Monarch 和 vLLM 的 Rollout Actor

## 1. 阶段目标

本阶段完成 agentic RL 原型的推理侧执行单元：使用 Monarch 管理 Rollout Actor，在 Actor 内通过 vLLM 部署策略模型，提供批量推理采样、Chat 采样、token 级 logprob、策略版本管理和运行时权重更新。

本阶段同时将共享配置重构为 Monarch、VeOmni 和 vLLM 三个独立区块，并验证训练配置与 rollout 配置可以共存。

本阶段不包含 RL 调度器、reward、advantage、GRPO 更新、trajectory 管理和上层 Agent Demo。

对应提交：

```text
a6a6e4e feat: add Monarch-based vllm rollout actor
```

## 2. 完成内容

主要新增或修改了以下文件：

```text
nano-agentic-rl/
├── actor/
│   ├── train_actor.py
│   └── rollout_actor.py
├── config/
│   └── qwen2_5_1_5b_gsm8k_sft.yaml
├── docs/
│   └── phase2_rollout_actor_implementation.md
├── test/
│   └── test_rollout_actor.py
└── main_sft.py
```

- `actor/rollout_actor.py`：实现基于 vLLM `AsyncLLM` 的 Monarch Rollout Actor。
- `test/test_rollout_actor.py`：验证启动、采样、Chat、权重更新和关闭流程。
- 配置文件：重构为三段式配置，并启用 vLLM 内部 DP。
- `actor/train_actor.py`：从共享配置中抽取 `veomni` 子树交给 VeOmni 原生解析器。
- `main_sft.py`：从 `monarch.train` 读取训练资源。

## 3. 最终架构

当前 Rollout Actor 使用一个 Monarch Actor 管理一个 vLLM 内部 DP deployment：

```text
Monarch Controller
    │
    └── RolloutActor
            │
            └── vLLM AsyncLLM
                    ├── DP Coordinator
                    ├── EngineCore DP rank 0
                    │   └── Worker: GPU 2
                    └── EngineCore DP rank 1
                        └── Worker: GPU 3
```

当前配置为：

```text
DP = 2
TP = 1
PP = 1
GPU 数量 = DP × TP × PP = 2
```

Monarch 负责 Rollout Actor 的生命周期，vLLM 负责 Actor 内部的 DP EngineCore、GPU worker、请求负载均衡和 DP Coordinator。

## 4. 三段式共享配置

配置文件顶层只保留三个区块：

```yaml
monarch:
  # 进程资源、GPU 分配、环境变量和关闭超时

veomni:
  # 数据适配、模型、数据和训练参数

vllm:
  # 推理引擎、并行策略和采样参数
```

### 4.1 Monarch 配置

```yaml
monarch:
  shutdown_timeout_seconds: 15
  env:
    NCCL_IB_DISABLE: "1"
  train:
    gpu_ids: [0, 1]
  rollout:
    gpu_ids: [2, 3]
```

GPU ID 属于资源控制层，因此统一放在 Monarch 配置中。训练和 rollout 使用独立的 GPU 列表，允许后续并发运行。

当前 CUDA 13.0 机器没有 IB，`NCCL_IB_DISABLE=1` 由所有 Monarch worker 继承。

### 4.2 VeOmni 配置

原阶段一的 `data_adapter`、`model`、`data` 和 `train` 已整体移动到 `veomni` 下。

VeOmni 原生解析器只认识其 dataclass 字段，不能直接解析完整三段式配置。因此 Train Actor 会：

1. 读取完整 YAML。
2. 提取 `veomni` 子树。
3. 移除框架自定义的 `data_adapter` 字段。
4. 写入临时 YAML。
5. 继续调用 VeOmni 公共 `parse_args` 接口。

该方式保留了 VeOmni 原生参数解析和校验逻辑。

### 4.3 vLLM 配置

```yaml
vllm:
  initial_policy_version: 0
  engine:
    data_parallel_size: 2
    data_parallel_backend: mp
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    model: /ssd/liuls/data/hub/Qwen2.5-1.5B-Instruct
    tokenizer: /ssd/liuls/data/hub/Qwen2.5-1.5B-Instruct
    dtype: bfloat16
    max_model_len: 1024
    gpu_memory_utilization: 0.5
    enforce_eager: true
    seed: 42
  sampling:
    n: 2
    max_tokens: 64
    temperature: 0.8
    top_p: 0.95
    logprobs: 1
```

`data_parallel_backend: mp` 表示 vLLM 使用本机多进程管理 DP ranks。`enforce_eager: true` 会关闭 `torch.compile` 和 CUDA Graph，适合当前原型调试及频繁权重更新。

Rollout Actor 在启动前校验：

```text
len(monarch.rollout.gpu_ids)
    == data_parallel_size × tensor_parallel_size × pipeline_parallel_size
```

配置不一致时会在加载模型前直接报错。

## 5. AsyncLLM 和内部 DP

早期实现使用同步 `vllm.LLM`。真实 DP 测试发现 vLLM 0.22.1 明确禁止：

```python
LLM(data_parallel_size=2)
```

同步 `LLM` 的单进程入口会抛出错误，以避免离线 DP 挂死。最终实现改为：

```python
AsyncLLM.from_engine_args(AsyncEngineArgs(...))
```

`AsyncLLM` 会启动统一前端、DP Coordinator 和多个 EngineCore，并在内部进行请求分发。Rollout Actor 的 endpoint 也因此改为异步接口。

## 6. Rollout 数据结构

Actor 返回普通 dataclass 和 Python 标量，避免将 vLLM 内部对象暴露到 Monarch RPC 边界。

### 6.1 RolloutSample

单条 completion 包含：

- `text`：生成文本。
- `token_ids`：response token IDs。
- `logprobs`：被选中 token 的逐 token log probability。
- `cumulative_logprob`：整条 completion 的累计 log probability。
- `finish_reason`：如 `stop` 或 `length`。
- `stop_reason`：命中的停止 token 或停止条件。

### 6.2 RolloutOutput

单个 prompt 的结果包含：

- 原始或 Chat Template 渲染后的 prompt。
- prompt token IDs。
- `n` 条 `RolloutSample`。
- 生成时使用的 `policy_version`。
- prefix cache 命中 token 数。

### 6.3 WeightUpdateResult

权重更新结果包含新策略版本、权重来源、tensor 数量和更新时间。

## 7. Rollout Actor 接口

### 7.1 setup

`setup` 完成以下工作：

1. 读取三段式配置。
2. 注入 Monarch 全局环境和 rollout GPU 可见列表。
3. 校验 DP、TP、PP 和 GPU 数量。
4. 创建 vLLM `AsyncEngineArgs`。
5. 启动 `AsyncLLM`、DP Coordinator、EngineCore 和 GPU workers。
6. 初始化策略版本。

### 7.2 generate

`generate` 支持单个 prompt 或 prompt 列表。

批量 prompts 会并发提交给 AsyncLLM，由 vLLM 内部 DP 分配到不同 engine core。文本在提交前经过 vLLM renderer，避免使用已弃用的 raw prompt 路径。

采样结果固定使用 `RequestOutputKind.FINAL_ONLY`，确保 `n > 1` 时最终返回全部 completions，而不是只保留流式输出的最后一个局部结果。

### 7.3 chat

`chat` 接收单个 conversation 或 conversation batch，使用模型 Chat Template 添加 generation prompt，再提交给 AsyncLLM。

### 7.4 update_weights

支持两种权重来源：

```text
checkpoint_path
HF state_dict
```

权重更新流程：

1. 获取 Actor 内部互斥锁，阻止新的 rollout 请求进入。
2. 校验新 policy version 必须大于当前版本。
3. 通过 `collective_rpc("reload_weights")` 向所有 vLLM workers 广播更新。
4. 清理所有 DP ranks 的 prefix cache。
5. 全部成功后提交新的 policy version。

state dict tensor 会先转换为 contiguous CPU tensor，再交给 vLLM 分片和加载。

### 7.5 状态和清理

- `get_status`：返回初始化状态、模型路径和策略版本。
- `reset_prefix_cache`：显式清理所有 DP ranks 的 prefix cache。
- `close`：取消 tokenizer 后台 batch tasks，关闭 AsyncLLM、EngineCore、DP Coordinator 和 workers，并清理 CUDA cache。

## 8. 并发与策略一致性

Actor 使用异步互斥锁保护生成和权重更新：

```text
generate/chat 执行中 -> update_weights 等待
update_weights 执行中 -> 新 generate/chat 等待
```

因此，一个 rollout 请求不会在生成过程中切换策略版本。权重更新 endpoint 返回后，后续请求才会使用新版本。

当前锁粒度覆盖整个 endpoint。单次 `generate` 内部的多个 prompts 仍会并发提交给 vLLM，但不同 Monarch endpoint 调用暂时不会相互并发。

## 9. 手动测试

运行：

```bash
conda activate rl
cd /ssd/liuls/projects/hw/nano-agentic-rl
python test/test_rollout_actor.py
```

可选参数：

```bash
python test/test_rollout_actor.py \
  --config config/qwen2_5_1_5b_gsm8k_sft.yaml \
  --checkpoint /ssd/liuls/data/hub/Qwen2.5-1.5B-Instruct \
  --version 1
```

测试内容：

1. 启动 Monarch Rollout Actor。
2. 启动 vLLM DP=2。
3. 并发提交两个 prompts。
4. 每个 prompt 生成两条 completion。
5. 检查 token IDs、logprobs 和 policy version。
6. 从本地 checkpoint 原地更新两个 DP workers。
7. 使用更新后的策略执行 Chat 推理。
8. 查询最终版本并关闭全部资源。

## 10. 已完成验证

### 10.1 配置兼容性

- YAML 顶层严格为 `monarch`、`veomni`、`vllm`。
- VeOmni 成功解析模型路径、global batch size 2 和 FSDP2。
- `main_sft.py` 正确读取训练 GPU 0/1。
- Rollout Actor 正确解析 DP=2、TP=1、PP=1 和 GPU 2/3。

### 10.2 vLLM 内部 DP

真实运行成功启动：

- 1 个 DP Coordinator。
- 2 个 EngineCore，分别为 DP rank 0 和 DP rank 1。
- 2 个 GPU worker，分别使用 GPU 2 和 GPU 3。

两个 prompt 均成功返回两条 completion，token IDs 和逐 token logprobs 完整。

### 10.3 权重更新

- 本地 Qwen checkpoint 同时加载到两个 DP workers。
- reload 时间约 1 秒。
- 两个 DP engine 的 prefix cache 均成功清理。
- policy version 从 0 更新到 1。
- 更新后 Chat 推理正常。

vLLM 在 Qwen checkpoint reload 时会输出：

```text
RotaryEmbedding: Failed to load weights
```

这是无独立持久权重的 RotaryEmbedding 产生的非致命 warning。更新后的推理已验证正常。

### 10.4 资源清理

- DP Coordinator、两个 EngineCore 和两个 workers 均正常退出。
- tokenizer 后台 tasks 在关闭前主动取消。
- GPU 2/3 最终显存恢复到约 1 MiB。
- Python bytecode 编译和 `git diff --check` 均通过。

## 11. 当前边界

- 目前只验证单机 vLLM `mp` DP，尚未验证多机 DP 或 Ray backend。
- 当前使用 checkpoint/state dict 更新，尚未接入 Train Actor 的自动权重传输通道。
- Actor endpoint 之间采用全局互斥，尚未开放多个 trajectory 的跨 endpoint 并发。
- 尚未实现 rollout queue、backpressure、优先级和超时控制。
- 尚未实现 reward、advantage、GRPO 和完整 RL 控制循环。
- 尚未定义 rollout trajectory、tool call 和 environment transition 数据模型。
- `enforce_eager: true` 以稳定和调试为先，尚未评估 CUDA Graph 模式下的性能和热更新行为。

至此，阶段二已经形成可独立运行的 Monarch + vLLM 推理侧基础单元，并验证了内部 DP、批量采样和多副本权重更新，为下一阶段连接 Train Actor 和实现 GRPO 调度提供了 rollout 执行层。

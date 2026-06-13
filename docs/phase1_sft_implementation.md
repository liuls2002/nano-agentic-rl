# 阶段一：基于 Monarch 和 VeOmni 的 SFT Train Actor

## 1. 阶段目标

本阶段完成了 agentic RL 原型的训练侧最小闭环：使用 Monarch 作为控制层，在单机多卡环境中启动 VeOmni 文本训练器，并对本地 Qwen2.5 模型和 GSM8K parquet 数据执行 SFT 训练。

本阶段只覆盖 Train Actor，不包含 vLLM Rollout Actor、训练与推理权重同步、GRPO 算法和上层 Agent Demo。

## 2. 完成内容

主要新增或完善了以下文件：

```text
nano-agentic-rl/
├── actor/
│   └── train_actor.py
├── config/
│   └── qwen2_5_1_5b_gsm8k_sft.yaml
├── docs/
│   └── phase1_sft_implementation.md
└── main_sft.py
```

- `actor/train_actor.py`：封装 VeOmni `TextTrainer`，实现 Monarch Train Actor。
- `main_sft.py`：负责 Monarch 进程启动、分布式环境初始化、Actor 调用和退出清理。
- `config/qwen2_5_1_5b_gsm8k_sft.yaml`：Qwen2.5-1.5B-Instruct 在 GSM8K 上的双卡 SFT 配置。

## 3. 总体架构

```text
main_sft.py（Monarch 控制器）
    │
    ├── 读取统一 YAML 配置
    ├── 创建单机多进程 ProcMesh
    ├── 向各进程注入 CUDA/NCCL 环境变量
    ├── 配置 Torch Elastic 分布式环境
    └── 在所有 rank 上启动 TrainActor
            │
            ├── 解析 VeOmni 参数
            ├── 创建 TextTrainer 或 GSM8K 数据适配 Trainer
            ├── 初始化模型、数据和 FSDP2
            └── 执行多卡 SFT 训练
```

数据处理链路如下：

```text
GSM8K parquet(question, answer)
    -> prompt_response 数据适配器
    -> system/user/assistant messages
    -> chat template 和 loss mask
    -> VeOmni DataLoader
    -> FSDP2 SFT
```

## 4. Train Actor 实现

`TrainActor` 是部署在 Monarch `ProcMesh` 上的 Actor，每个 GPU 进程对应一个 Actor 实例。

### 4.1 setup

`setup` endpoint 完成以下工作：

1. 读取原始 YAML 配置。
2. 在 Monarch 已设置好 rank 环境后，调用 VeOmni 公共 `parse_args` 接口解析配置。
3. 根据 `data_adapter.type` 选择原生 `TextTrainer` 或 `PromptResponseTextTrainer`。
4. 初始化 VeOmni 模型、数据集、优化器和分布式训练环境。

VeOmni 的配置解析依赖 `sys.argv`。实现中只在解析配置的短暂区间替换参数，并在结束后恢复，避免持续污染 Actor 进程的命令行状态。

### 4.2 train

`train` endpoint 调用 VeOmni Trainer 的完整训练流程。正常情况下，VeOmni 负责训练生命周期和分布式进程组销毁；Actor 额外在 `finally` 中检查并清理仍存活的 Torch distributed process group，作为异常路径的兜底。

### 4.3 GSM8K 数据适配

GSM8K parquet 使用独立的 `question` 和 `answer` 字段，而 VeOmni 原生 conversation 数据流期望 messages 格式。为此增加了轻量适配层，将单条样本转换为：

```text
system    loss_mask=0
user      loss_mask=0
assistant loss_mask=1
```

只有 assistant answer 参与 loss 计算，system prompt 和用户问题均被 mask。字段名和 system prompt 都可通过配置修改：

```yaml
data_adapter:
  type: prompt_response
  prompt_key: question
  response_key: answer
  system_prompt: Solve the math problem step by step and give the final answer.
```

## 5. Monarch 多卡启动

控制器参考 torchforge-ascend 的设计，使用普通进程 `ProcMesh` 启动每个训练 rank，再通过 `setup_torch_elastic_env_async` 建立 Torch Elastic 所需环境。

启动顺序为：

1. 创建包含 `num_gpus` 个进程的 `ProcMesh`。
2. 使用 `EnvSetter` Actor 向所有进程写入 `CUDA_VISIBLE_DEVICES` 和额外环境变量。
3. 配置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、master 地址和端口。
4. 在整个 mesh 上创建 `TrainActor`。
5. 广播执行 `setup` 和 `train` endpoint。

没有直接使用 Monarch GPU mesh，是因为早期测试中其设备映射与 VeOmni/NCCL 的 rank 设备选择存在歧义，在 NCCL barrier 阶段出现过 `SIGSEGV`。当前方案明确共享可见 GPU 列表，并由 elastic local rank 选择设备，双卡训练已验证通过。

## 6. 配置说明

测试配置位于：

```text
config/qwen2_5_1_5b_gsm8k_sft.yaml
```

关键路径：

```yaml
model:
  model_path: /ssd/liuls/data/hub/Qwen2.5-1.5B-Instruct
  tokenizer_path: /ssd/liuls/data/hub/Qwen2.5-1.5B-Instruct

data:
  data_path: /ssd/liuls/data/hub/gsm8k/main/train-00000-of-00001.parquet
```

当前默认使用 2 张 GPU、FSDP2、BF16、单卡 micro batch size 1、global batch size 2，训练上限为 100 steps。

CUDA 13.0 环境下当前机器没有 IB，因此通过 Monarch 配置统一向 worker 注入：

```yaml
monarch:
  num_gpus: 2
  gpu_ids: [0, 1]
  shutdown_timeout_seconds: 15
  env:
    NCCL_IB_DISABLE: "1"
```

该变量属于当前机器环境约束，而不是 Train Actor 的硬编码行为，后续可直接通过配置调整。

模型 attention backend 当前配置为 PyTorch SDPA，因此不依赖 `flash_attn`。若切换到 `flash_attention_2`，运行环境需要安装对应版本的 `flash_attn`。

## 7. 中断和错误管理

针对训练期间按一次 `Ctrl+C` 无法顺利退出的问题，控制器增加了统一的 Monarch 生命周期管理。

### 7.1 信号处理

- 捕获 `SIGINT` 和 `SIGTERM`，第一次信号触发异步 shutdown event。
- 正在等待的 Actor endpoint 调用会被取消等待，控制器立即进入清理阶段。
- 第二次 `Ctrl+C` 保留强制中断能力，避免异常情况下无限等待。
- 主进程按惯例返回退出码 `130`（SIGINT）或 `143`（SIGTERM）。

### 7.2 Monarch 故障处理

- 注册 Monarch unhandled fault hook，集中接收 mesh/worker 故障。
- 正常 Actor 异常保留原始 Monarch `ActorError`，不会被错误转换成 `KeyboardInterrupt`。
- worker 异常、用户中断和正常结束最终都进入同一清理路径。

### 7.3 有界清理

退出时先停止 `ProcMesh`，再调用 Monarch `shutdown_context`。整个过程受 `shutdown_timeout_seconds` 约束，避免控制器永久卡在资源回收阶段。

在 CUDA kernel 或同步训练步骤内部中断时，Monarch 需要直接终止 worker 进程，底层偶尔会输出 `terminate called without an active exception`。这不影响最终清理结果；测试中一次 `Ctrl+C` 即可退出，未残留训练 worker 或 GPU 占用。

## 8. 运行方式

```bash
conda activate rl
cd /ssd/liuls/projects/hw/nano-agentic-rl
python main_sft.py --config config/qwen2_5_1_5b_gsm8k_sft.yaml
```

`NCCL_IB_DISABLE=1` 已包含在 YAML 配置中，不需要在上述命令前重复 export。

## 9. 已完成验证

### 9.1 静态检查

- `actor/train_actor.py` 和 `main_sft.py` 通过 Python bytecode 编译检查。
- VeOmni 配置在模拟双卡 rank 环境下成功解析。
- 解析结果为 `dp_size=2`、`dp_shard_size=2`、global batch size 2、gradient accumulation 1。

### 9.2 数据检查

使用真实 GSM8K parquet 首条样本和本地 Qwen2.5 tokenizer 验证：

- 成功生成 `input_ids`、`attention_mask` 和 `labels`。
- 测试样本序列长度为 113 tokens。
- assistant 有效监督部分为 62 tokens。
- system prompt 和 user question 共 51 tokens，被正确 mask。

### 9.3 双卡训练

使用 GPU 0 和 GPU 1 上的两张 RTX 3090 完成了 2-step smoke test：

- 两个 rank 均成功初始化 FSDP2。
- loss 从约 0.87 下降到约 0.33。
- 单卡峰值显存约 13.07 GiB。
- 训练结束后 Monarch 和 Torch distributed 均正常退出。

这里的 2 steps 是阶段验证时使用的临时 smoke test 参数；当前提交的配置已设置为 `max_steps: 100`。

### 9.4 中断和异常

- Actor setup 阶段按一次 `Ctrl+C`，约 3.4 秒退出，返回码 130。
- 活跃训练阶段按一次 `Ctrl+C`，约 1.8 秒退出，返回码 130。
- 两种情况均未残留 Monarch worker 或 GPU 0/1 上的训练进程。
- 使用缺失 `flash_attn` 的配置触发真实初始化错误时，错误以 Monarch Actor traceback 正常返回，并完成资源清理。

## 10. 当前边界

本阶段实现仍有以下明确边界：

- 只支持当前验证过的单机多卡训练流程，尚未验证多机训练。
- 只实现 SFT Train Actor，没有 Rollout Actor 和权重同步。
- 尚未实现 advantage、reward、GRPO 更新等 RL 组件。
- `prompt_response` 适配器面向独立问题/答案列，尚未扩展为通用 agent trajectory 数据格式。
- 活跃 CUDA step 中断采用进程级停止，不是训练 step 边界上的协作式取消。
- 尚未构建 SQL Agent 和 Claude Code Agent 训练 Demo。

至此，阶段一已经形成可独立运行和验证的 Monarch + VeOmni 多卡 SFT 基础，为后续接入 vLLM Rollout Actor 和 RL 调度流程提供训练侧执行单元。

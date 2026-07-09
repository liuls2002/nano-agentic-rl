# Nano-Agentic-RL
Nano-Agentic-RL 是一个基于 Monarch + VeOmni + vLLM 搭建的一个 agentic RL 训练框架原型，一个轻量、可扩展的框架，支持SFT、同步/异步 RL 训练。
- 训练后端 VeOmni：https://github.com/ByteDance-Seed/VeOmni
- 推理后端 vLLM：https://github.com/vllm-project/vllm
- 控制层 Monarch：https://github.com/meta-pytorch/monarch

## Overview

Nano-Agentic-RL 关注的是训练系统原型，而不是单个算法脚本。项目把数据、训练、推理、奖励、优势计算、replay buffer 和评估都抽象成 Monarch Actor，由 controller 用普通 Python 代码编排完整训练流程。

```text
                    Monarch Controller
                            |
        +-------------------+-------------------+
        |                   |                   |
  support actors       TrainActor mesh      RolloutActor
        |                   |                   |
 Dataset / Reward      VeOmni + FSDP2       vLLM AsyncLLM
 Advantage / Replay    optimizer step       rollout / eval
```

这个设计让训练侧和推理侧保持低耦合：VeOmni 专注模型训练，vLLM 专注高吞吐采样，Monarch 专注分布式进程、Actor 消息和控制流。SFT、同步 RL、异步 RL 共享同一批组件，只替换 controller 的调度策略。

当前实验结果和曲线汇总见 [实验结果分析](docs/experiments.md)：在 GSM8K 实验中，RL 明显强于 SFT，异步 GRPO 在接近同步 GRPO 效果的同时显著缩短 wall-clock。

## Design Philosophy

Nano-Agentic-RL 的核心设计是 **everything as an actor**。

Monarch 的 Actor 是带私有状态的远程计算单元，通过 endpoint 接收异步消息；多个 Actor 可以组成 mesh，controller 可以对 mesh 广播调用，也可以对单个 Actor 发起 RPC。这非常适合 agentic RL 系统，因为 RL 训练天然包含多个有状态服务：训练器、推理引擎、数据流、reward、replay buffer 和评估器。

当前架构遵循几条原则：

- **控制逻辑显式**：训练流程写在 `main_sft.py`、`main_rl.py`、`main_rl_async.py` 中，不隐藏在重型 trainer 或服务框架里。
- **状态边界清晰**：模型权重属于 `TrainActor` 和 `RolloutActor`，样本队列属于 `ReplayBufferActor`，数据游标属于 `DatasetActor`。
- **后端低侵入**：不重写 VeOmni 的 FSDP、optimizer、scheduler，也不重写 vLLM 的调度和采样，只在 Actor 边界做封装。
- **资源隔离明确**：训练 GPU 和推理 GPU 由 controller 按 `CUDA_VISIBLE_DEVICES` 顺序切分，方便单机多卡调试。
- **同步机制可控**：controller 明确决定何时 rollout、何时 train、何时同步权重、何时 eval。
- **易于切换同步/异步**：同步 RL 和异步 RL 使用相同 Actor，只改变 replay buffer 和 rollout producer/trainer consumer 的协同方式。

这种设计借鉴了 [meta-pytorch/torchforge](https://github.com/meta-pytorch/torchforge) 的思想：把基础设施关注点和算法关注点分开。算法实验主要关注 reward、advantage、sampling、loss 和数据流；进程管理、RPC、资源生命周期和错误传播交给 Monarch。

## Architecture

当前主要组件如下：

| Component | Backend | Responsibility |
| --- | --- | --- |
| `DatasetActor` | Python / tokenizer | 加载预处理 jsonl，提供 train/eval batch |
| `TrainActor` | VeOmni | stepwise SFT / GRPO 训练，FSDP2，optimizer，scheduler |
| `RolloutActor` | vLLM | rollout 采样、eval 采样、在线权重接收 |
| `RewardActor` | Python rules | 答案抽取、正确性奖励、格式奖励 |
| `AdvantageActor` | Python | group reward 标准化，低方差判断 |
| `ReplayBufferActor` | Python / torch tensors | 存储 episode，按 rank 构造训练 batch |
| Controller | Monarch | 启动 Actor，编排训练、eval、权重同步和 W&B |

权重同步使用 vLLM 的 NCCL weight transfer。trainer rank0 发送 FSDP all-gather 后的完整参数，vLLM workers 直接接收并更新 policy version。异步 RL 中，如果同步发生时 vLLM 正在生成，`RolloutActor` 会调用 `pause_generation(mode="abort")` 中断 in-flight request，controller 再将整批样本放回 retry queue。

## Training Modes

### SFT

`main_sft.py` 使用 `DatasetActor` 提供训练样本，`TrainActor` 执行 VeOmni stepwise SFT，`RolloutActor` 只用于评估。流程包括 baseline eval、按 step 训练、间隔权重同步和 eval，以及 final eval。

```bash
python main_sft.py config/qwen3_1_7b_gsm8k_sft.yaml
```

### 同步 GRPO

`main_rl.py` 实现同步 RL 闭环。每个 step 严格执行：

```text
rollout -> reward -> advantage -> replay batch -> train -> weight sync -> eval
```

同步模式最容易理解和调试，适合验证 reward、advantage、GRPO loss、权重同步和 eval 指标是否正确。

```bash
python main_rl.py config/qwen3_1_7b_gsm8k_grpo.yaml
```

### 异步 GRPO

`main_rl_async.py` 实现 producer-consumer 异步 RL：

```text
rollout producer: dataset -> vLLM -> reward -> advantage -> replay buffer
trainer consumer: replay buffer -> GRPO train step -> weight sync
```

异步模式允许 rollout 和 train 重叠。`rl.replay_buffer.max_policy_age` 控制 trainer 可以使用多旧的样本；例如 `max_policy_age: 1` 表示允许使用上一版本 policy 生成的样本。权重同步会中断当前 vLLM generation，被中断的整批 prompt 会进入 retry queue，并在下一轮用新 policy 重新生成。

```bash
python main_rl_async.py config/qwen3_1_7b_gsm8k_grpo_async.yaml
```

## 配置结构

当前配置使用统一的新结构：

- `monarch`：全局控制配置、环境变量、序列长度、W&B。
- `train_actor`：VeOmni model/train 配置和训练 GPU 数。
- `rollout_actor`：vLLM engine、rollout/eval sampling 配置和推理 GPU 数。
- `dataloader`：预处理后的 train/eval jsonl 数据。
- `rl`：GRPO、reward、advantage、replay buffer、weight sync 配置。

GPU 分配由 controller 根据 `CUDA_VISIBLE_DEVICES` 和 actor GPU 数顺序切分：train actor 使用前 N 张可见 GPU，rollout actor 使用后续 GPU。

## Quick Start

环境假设：

- 已安装当前仓库、Monarch、VeOmni、vLLM、torch。
- 使用 `conda activate rl`。
- 模型和数据路径已经写入对应 YAML 配置。

完整环境配置见 [安装说明](docs/install.md)。当前推荐环境固定使用 `VeOmni v0.1.11`、`torchmonarch==0.5.0` 和 `vllm==0.22.1`。

示例：

```bash
cd /ssd/liuls/projects/hw/nano-agentic-rl
conda activate rl

# SFT
python main_sft.py config/qwen3_1_7b_gsm8k_sft.yaml

# Synchronous GRPO
python main_rl.py config/qwen3_1_7b_gsm8k_grpo.yaml

# Asynchronous GRPO
python main_rl_async.py config/qwen3_1_7b_gsm8k_grpo_async.yaml
```

如果只希望使用部分 GPU，可以通过 `CUDA_VISIBLE_DEVICES` 控制可见设备。controller 会在可见 GPU 池内顺序分配 train 和 rollout 资源。

## 文档

- [安装说明](docs/install.md)
- [Actor 设计](docs/actors.md)
- [权重同步](docs/weight_sync.md)
- [异步 RL 工作流程](docs/async_rl.md)
- [实验结果分析](docs/experiments.md)

## 致谢

Nano-Agentic-RL 建立在以下开源项目和设计思想之上：

- [meta-pytorch/monarch](https://github.com/meta-pytorch/monarch)：分布式控制层与 Actor 编排能力。
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：高吞吐 LLM 推理、采样和在线权重更新能力。
- [ByteDance-Seed/VeOmni](https://github.com/ByteDance-Seed/VeOmni)：训练后端、FSDP2、optimizer/scheduler 和模型训练组件。
- [meta-pytorch/torchforge](https://github.com/meta-pytorch/torchforge)：Actor 化训练系统与控制/计算解耦的设计参考。
- [THUDM/slime](https://github.com/THUDM/slime)：RL 训练流程、GRPO/DAPO 实验配置和系统设计参考。

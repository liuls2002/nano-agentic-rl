# 权重同步

本文介绍 VeOmni trainer 到 vLLM rollout actor 的权重同步设计。当前同步路径使用 vLLM 的 NCCL weight transfer：训练侧 rank0 发送 FSDP all-gather 后的完整参数，vLLM workers 直接接收并更新推理权重。

## 为什么需要在线权重同步

RL 训练中，trainer 每完成一个 optimizer step 后，rollout actor 应该使用最新 policy 继续采样。直接让 vLLM 从 checkpoint 重新加载太慢，也会引入大量磁盘 IO。因此当前实现使用 GPU 间 NCCL 直传：

```text
VeOmni/FSDP2 trainer ranks
  -> all-gather full parameters
  -> trainer rank0 NCCL send
  -> vLLM workers NCCL receive
  -> update policy version
```

SFT 也复用同一套同步能力，用于训练前 baseline eval 和训练中 interval eval。

## 同步入口

controller 调用公共 helper：

```python
sync_policy_weights(
    rollout_actor=rollout_actor,
    train_actors=train_actors,
    weight_metadata=weight_metadata,
    version=policy_version + 1,
)
```

内部并发执行：

```python
rollout_actor.receive_weights.call_one(...)
train_actors.broadcast_weights.call()
```

这两个调用必须同时发起：vLLM 侧先进入接收状态，trainer 侧随后广播参数，NCCL group 才能完成传输。

## Metadata

`TrainActor.get_weight_transfer_metadata()` 返回：

```python
{
    "names": [...],
    "dtype_names": [...],
    "shapes": [...]
}
```

它描述本次要传给 vLLM 的参数顺序、shape 和传输 dtype，不包含 tensor 内容。

虽然该 endpoint 会在所有 train rank 上执行，但每个 rank 返回的是逻辑全模型 metadata。controller 当前取其中一份传给 rollout actor。真正参数传输发生在 `broadcast_weights()`。

## Trainer 侧流程

`TrainActor.init_weight_transfer()` 初始化 trainer 侧 NCCL group。只有训练 rank0 持有发送 engine，其他 FSDP ranks 参与参数 all-gather。

`TrainActor.broadcast_weights()` 执行：

1. 遍历 `trainer.base.model.named_parameters()`。
2. 对 FSDP2/DTensor 参数调用 `full_tensor()`，所有 rank 共同参与 collective。
3. rank0 得到完整参数。
4. 浮点参数 cast 到 rollout engine dtype。
5. rank0 通过 vLLM `NCCLWeightTransferEngine.trainer_send_weights()` 发送。
6. 非 rank0 只参与 `full_tensor()`，不发送。

这保证 vLLM 收到的是 HF-format 的完整参数，而不是某个 FSDP shard。

## vLLM 侧流程

`RolloutActor.init_weight_transfer()` 让 vLLM workers 加入 NCCL transfer group。

`RolloutActor.receive_weights()` 执行：

1. 关闭 `_generation_gate`，阻止新的 generation 进入。
2. 获取 actor `_lock`，避免多个权重更新并发。
3. 调用 `llm.pause_generation(mode="abort")`。
4. 调用 `llm.start_weight_update(is_checkpoint_format=True)`。
5. 调用 `llm.update_weights(...)` 接收 NCCL 参数。
6. 调用 `llm.finish_weight_update()`。
7. reset prefix cache。
8. 更新 `policy_version`。
9. 调用 `llm.resume_generation()`。
10. 重新打开 `_generation_gate`。

`pause_generation(mode="abort")` 会中止 vLLM 内所有未完成请求，并让 scheduler 进入 paused 状态。`resume_generation()` 只负责重新打开调度，不会恢复已经 abort 的请求。

## dtype 语义

metadata 中的 dtype 是传输/推理 dtype，来自：

```yaml
rollout_actor:
  engine:
    dtype: bfloat16
```

当前混合精度训练中，VeOmni/FSDP2 的常驻主参数通常是 fp32，FSDP forward/backward 使用 bf16 参数视图。同步给 vLLM 时，浮点参数会 cast 到 `rollout_actor.engine.dtype`，所以 vLLM 使用 bf16 权重推理。

初始权重同步中常见路径是：

```text
checkpoint bf16 -> trainer fp32 -> vLLM bf16
```

`bf16 -> fp32` 是精确扩展，`fp32 -> bf16` 对原本来自 bf16 的值通常是稳定 round-trip。训练若已经更新 fp32 主权重，则同步到 vLLM 时会正常量化到 bf16，这是 mixed precision 训练和 bf16 推理的预期边界。

## policy version

rollout actor 持有 `policy_version`。同步成功后，`receive_weights()` 将其更新到 controller 指定的版本。

同步版 RL 中，每个 train step 后同步一次：

```text
v0 baseline
step 1 -> sync -> v1
step 2 -> sync -> v2
```

异步版 RL 中，replay buffer 依赖 episode 的 `policy_version` 和 `rl.replay_buffer.max_policy_age` 控制样本新鲜度。`max_policy_age=1` 表示当前 policy 可以训练上一版本和当前版本的样本。

## cache 和 generation 中断

权重更新后必须清理 prefix cache，否则 cache 中可能包含旧权重下的中间状态。当前同步完成后会调用 `reset_prefix_cache()`。

异步 RL 中，如果 rollout 正在生成，权重同步会通过 `pause_generation(mode="abort")` 中断 in-flight request。被中断的请求不会由 vLLM 自动恢复；controller 会捕获 interruption，把整批原始 dataset samples 放回 retry queue，后续使用新 policy 重新生成。

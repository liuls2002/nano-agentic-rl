# 异步 RL 工作流程

本文介绍 `main_rl_async.py` 中的异步 GRPO 控制流。它的目标是让 rollout 和 train 尽量重叠：rollout producer 持续生成样本写入 replay buffer，trainer consumer 持续从 replay buffer 取样训练。

## 总体流程

异步 RL 启动流程和同步 RL 类似：

1. 启动 train/eval `DatasetActor`。
2. 启动 `RewardActor`、`AdvantageActor`、`ReplayBufferActor`。
3. 启动 VeOmni `TrainActor` mesh。
4. 启动 vLLM `RolloutActor`。
5. 初始化 NCCL weight transfer。
6. trainer 初始权重同步到 vLLM，保持 baseline policy 一致。
7. 执行 baseline eval。
8. 启动 rollout producer task。
9. trainer consumer 开始训练主循环。

主循环由两个并发角色组成：

```text
rollout producer:
  dataset -> vLLM rollout -> reward -> advantage -> replay buffer

trainer consumer:
  replay buffer -> TrainActor.train_grpo_step -> weight sync -> eval
```

## Rollout producer

producer 持续执行一轮 rollout iteration：

1. 等待 `rollout_gate` 打开。
2. 如果 replay buffer 接近满容量，则短暂 sleep。
3. 优先从 `retry_queue` 取样本。
4. 不够时从 `train_dataset.next_batch()` 取新样本。
5. 调用 `RolloutActor.chat()` 采样。
6. 过滤超长 prompt/response 和低方差 group。
7. 使用 `RewardActor` 计算 reward。
8. 使用 `AdvantageActor` 计算 group advantage。
9. 构造 `RLEpisode` 并写入 replay buffer。

每轮请求的 group 数由：

```text
ceil(rl.rollout_batch_size * rl.rollout_batch_size_multiplier)
```

和 `rl.max_rollout_groups_per_step` 共同限制。

## Trainer consumer

trainer 主循环由 `rl.max_steps` 控制。每一步：

1. 调用 `replay_buffer.sample(current_policy_version)`。
2. 如果 buffer 不足，sleep 并累计 `async/train_wait_buffer_sec`。
3. 拿到 batch 后执行 `train_actors.train_grpo_step.call(batches)`。
4. 关闭 `rollout_gate`，阻止新的 rollout iteration 开始。
5. 设置 `sync_in_progress`。
6. 调用 NCCL weight sync，将 trainer 最新权重同步给 vLLM。
7. 清除 `sync_in_progress`。
8. 如到达 eval interval，暂停 rollout 并执行 eval。
9. 重新打开 `rollout_gate`。

trainer 不等待 rollout producer 自然结束。若 producer 正在 vLLM 生成，权重同步会主动中断该生成。

## Replay buffer 和 policy age

replay buffer 按 episode 存储。一个 prompt group 通常对应 `sampling.n` 条 episode，但 buffer 的 sample 和 consume 都是 episode-level。

`max_policy_age` 控制可训练样本的新鲜度：

```text
current_policy_version - episode.policy_version <= max_policy_age
```

如果 `max_policy_age=0`，只训练当前 policy 样本，异步 overlap 会很弱。如果 `max_policy_age=1`，trainer 可以使用上一版本和当前版本的样本，通常更适合异步 RL。

`consume_samples: true` 时，被 trainer 采样出的 episode 会从 buffer 删除。由于过期清理和随机消费都是 episode-level，日志中的 buffer size 不保证是 `sampling.n` 的倍数。

## Gate 和事件

异步协同依赖四个控制对象：

- `rollout_gate`：controller 侧事件。关闭后，producer 不会开始新的 rollout iteration。
- `sync_in_progress`：controller 侧事件。标记当前 interruption 是否由权重同步触发。
- `retry_queue`：controller 侧队列。保存被中断后需要整批重试的 `DatasetSample`。
- `_generation_gate`：RolloutActor 内部事件。关闭后，新的 `chat()`/`generate()` 会等待权重更新完成。

两层 gate 的分工不同：

```text
rollout_gate:
  控制 producer 是否开始下一轮 rollout

_generation_gate:
  控制 RolloutActor 是否允许新的 generation 进入 vLLM
```

## in-flight generation 中断

`RolloutActor.chat()` 在准备 prompt、sampling params 和 policy version 时短暂持有 `_lock`。真正调用 vLLM generate 时不持有 `_lock`。

这样做是为了让权重更新 endpoint 能在 generation 过程中进入 actor，并调用：

```python
await llm.pause_generation(mode="abort")
```

`abort` 会：

- 将未完成 request 标记为 aborted。
- 从 vLLM scheduler running/waiting 队列移除。
- 释放 KV/cache。
- 给这些 request 返回 `finish_reason="abort"` 的 final output。

已经正常 finished 的 request 不会被改成 abort；但当前 controller 不消费部分完成结果。一旦本轮 rollout 被权重同步打断，整轮被视为 interrupted。

## 整批重试

producer 每轮先取出一个 `dataset_samples` batch：

```text
[g1, g2, g3, ...]
```

如果 `collect_rollout_iteration()` 因权重同步中断抛异常，producer 会执行：

```python
retry_rollout_samples(retry_queue, dataset_samples)
```

下轮 rollout 优先从 `retry_queue` 取这些样本，用新 policy 重新生成。

当前选择整批重试，而不是部分重试，原因是实现更简单、policy version 更清晰、不会把半批旧 policy 结果混入 replay buffer。代价是可能浪费已经完成但尚未被 controller 处理的 response。

## 同步 RL 与异步 RL 对比

同步 RL：

- 每一步严格按 rollout -> train -> sync 执行。
- on-policy 语义最清晰。
- 更适合调试算法和定位问题。
- trainer 和 rollout 不能重叠，GPU 利用率可能较低。

异步 RL：

- rollout producer 和 trainer consumer 并行。
- 可以允许 `max_policy_age > 0` 的轻微 stale sample。
- 训练结束一步后立即中断 in-flight rollout 并同步权重。
- 需要处理 retry queue、policy age、buffer 等工程细节。

当前异步实现是系统原型中更接近高效 RL 训练的路径，后续可以在此基础上继续优化为部分重试、更多 rollout actor、多环境 agent trajectory 或更细粒度的 policy version 管理。

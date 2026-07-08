结论：**RL 最终训练步数由 `rl.max_steps` 控制；但学习率 scheduler 的总步数不是由 `rl.max_steps` 控制，而是由 VeOmni 根据训练数据集和 `train_actor.train.num_train_epochs/max_steps` 算出来。当前能工作，但不完全适配 RL 语义。**

**谁控制 RL 最终步数**
`main_rl.py` 里外层循环是：

```python
for step in range(1, settings["max_steps"] + 1):
```

而 `settings["max_steps"]` 来自：

```python
max_steps = int(rl_config.get("max_steps", 1))
```

也就是 [main_rl.py](/ssd/liuls/projects/hw/nano-agentic-rl/main_rl.py:229) 和 [main_rl.py](/ssd/liuls/projects/hw/nano-agentic-rl/main_rl.py:1104)。

所以当前配置：

```yaml
rl:
  max_steps: 1000
```

最终就是 controller 跑 1000 个 RL step。每个 RL step 做一次：

1. rollout + dynamic sampling
2. replay batch
3. `train_actors.train_grpo_step.call(batches)`
4. weight sync
5. 间隔 eval

`train_actor.train.max_steps` 不控制这个外层 RL 循环。

**VeOmni scheduler 是否会正常 step**
会。`TrainActor.train_grpo_step()` 里每次 GRPO update 都会：

```python
base.optimizer.step()
base.lr_scheduler.step()
```

见 [actor/train_actor.py](/ssd/liuls/projects/hw/nano-agentic-rl/actor/train_actor.py:456)。

所以 warmup / decay 机制本身是生效的，日志里的 `lr_mean` 也是从 optimizer param group 读出来的真实当前 lr。

**但 scheduler 的总步数不等于 RL max_steps**
VeOmni 构造 scheduler 时用的是：

```python
train_steps=args.train_steps * args.train.num_train_epochs
```

见 [base.py](/ssd/liuls/projects/hw/VeOmni/veomni/trainer/base.py:537)。

这里的 `args.train_steps` 来自 VeOmni 数据集长度，不来自 `rl.max_steps`。对 mapping dataset，它还会先除以 DP size：

```python
dataset_length = dataset_length / args.train.accelerator.dp_size
args.compute_train_steps(dataset_length)
```

见 [base.py](/ssd/liuls/projects/hw/VeOmni/veomni/trainer/base.py:448)。

当前配置大概是：

```yaml
dataloader.train.path: data/train/gsm8k_train.jsonl  # 7473 条
train_actor.num_gpus: 4
global_batch_size: 32
dataloader_batch_size per rank: 32 / 4 = 8
num_train_epochs: 10
train.max_steps: 1000
```

所以 VeOmni 每 epoch 的 train_steps 约为：

```text
floor((7473 / 4) / 8) = 233
```

scheduler 总步数约为：

```text
233 * 10 = 2330
```

而 RL controller 实际只跑：

```text
rl.max_steps = 1000
```

所以当前 scheduler 看到的是 2330-step 训练计划，但实际只执行 1000 个 GRPO optimizer step。

**当前配置下影响大吗**
你现在是：

```yaml
lr: 1.0e-6
lr_warmup_ratio: 0.0
lr_decay_style: constant
lr_decay_ratio: 1.0
```

所以当前实际行为是：

```text
从第 1 个 GRPO step 开始就是 1e-6
没有 warmup
没有 decay
1000 step 全程基本保持 1e-6
```

这个配置下，scheduler 总步数不匹配影响很小，因为 constant + no warmup 根本不依赖总步数。

**如果你加 warmup/decay，就会有适配问题**
例如你想：

```yaml
lr_warmup_ratio: 0.01
```

你可能以为是：

```text
1000 * 0.01 = 10 warmup steps
```

但实际是：

```text
2330 * 0.01 = 23 warmup steps
```

如果设置：

```yaml
lr_decay_style: cosine
```

cosine decay 也会按 2330 步规划；但 controller 只跑到 1000 步，所以训练结束时 lr 还没有按你预期 decay 到对应位置。

**推荐做法**
短期如果不改代码：

```yaml
optimizer:
  lr: 1.0e-6
  lr_warmup_ratio: 0.0043  # 约等于 10 / 2330
  lr_decay_style: constant
```

这样可以得到大约 10 step warmup。

更合理的工程修复是：**RL 模式下让 VeOmni scheduler 的 total steps 直接使用 `rl.max_steps`**，不要再用 SFT dataloader 的 `train_steps * num_train_epochs`。否则 RL 的学习率配置总是间接依赖 train jsonl 长度、DP size、`num_train_epochs`，不够直观。

所以回答你的问题：

- 最终 RL 步数：`rl.max_steps`
- 当前 LR/warmup/decay 是否能工作：能工作
- 当前是否适配 RL：`constant + warmup=0` 没问题；一旦加 warmup/decay，当前总步数计算不完全适配，建议改成按 `rl.max_steps` 构造 scheduler。
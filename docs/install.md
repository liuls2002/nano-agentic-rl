# 安装说明

本文档记录 Nano-Agentic-RL 当前实验环境的推荐安装方式。示例环境基于 CUDA 13.0、Python 3.12 和 conda，默认环境名为 `rl`。

## 前置条件

- 已安装 NVIDIA Driver，并能正常执行 `nvidia-smi`。
- 已安装 conda 或 miniconda。
- 当前机器可以访问 PyPI 和 GitHub。
- 建议预留独立目录存放外部依赖仓库，例如 `/ssd/liuls/projects/hw`。

## 创建 Python 环境

```bash
conda create -n rl python=3.12
conda activate rl

pip install -U pip
pip install uv
```

后续命令默认在 `rl` 环境中执行。

## 安装 Monarch 和 vLLM

```bash
uv pip install torchmonarch==0.5.0 --python "$CONDA_PREFIX/bin/python"
uv pip install vllm==0.22.1 --python "$CONDA_PREFIX/bin/python"
```

当前实验使用 `torchmonarch==0.5.0` 和 `vllm==0.22.1`。如果后续升级这两个依赖，需要重新验证 Actor RPC、vLLM AsyncLLM、在线权重更新和 `pause_generation(mode="abort")` 行为。

## 安装 VeOmni

VeOmni 固定使用 `v0.1.11`，避免训练 runtime、FSDP2、optimizer/scheduler 或模型 patch 行为随上游变化。

```bash
cd /ssd/liuls/projects/hw

git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
git checkout v0.1.11

pip install -e .
```

如果本地已经存在 VeOmni 仓库，可以直接切换到固定版本后重新安装：

```bash
cd /ssd/liuls/projects/hw/VeOmni
git fetch --tags
git checkout v0.1.11
pip install -e .
```

## 安装 Nano-Agentic-RL

```bash
cd /ssd/liuls/projects/hw/nano-agentic-rl
pip install -e .
```

如果当前仓库没有声明完整的 Python package metadata，也可以直接在仓库根目录运行脚本；核心要求是 `monarch`、`vllm`、`torch`、`VeOmni` 能在同一个 `rl` 环境中 import。

## 验证安装

```bash
python - <<'PY'
import torch
import monarch
import vllm
import veomni

print("torch:", torch.__version__)
print("monarch:", getattr(monarch, "__version__", "unknown"))
print("vllm:", vllm.__version__)
print("veomni:", getattr(veomni, "__version__", "unknown"))
print("cuda available:", torch.cuda.is_available())
PY
```

也可以运行静态检查：

```bash
python -m py_compile main_sft.py main_rl.py main_rl_async.py actor/*.py tools/*.py
```

## 运行示例

```bash
# SFT
python main_sft.py config/qwen3_1_7b_gsm8k_sft.yaml

# Synchronous GRPO
python main_rl.py config/qwen3_1_7b_gsm8k_grpo.yaml

# Asynchronous GRPO
python main_rl_async.py config/qwen3_1_7b_gsm8k_grpo_async.yaml
```

如果只希望使用部分 GPU，可以通过 `CUDA_VISIBLE_DEVICES` 控制可见设备。controller 会在可见 GPU 池内顺序分配 train actor 和 rollout actor 资源。

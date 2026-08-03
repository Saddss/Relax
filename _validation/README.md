# 多卡验证计划 —— 任务 28 RLOO（review 修复后）

**一句话**：本次改动把 RLOO 的归约改成 completion-level（梯度分母从 token 数换成 sample 数），新增 `get_cp_local_num_samples`。这个新分母**只有单元测试覆盖**，没有跑过真实 CP 组的 all-reduce。这次要验的就是它。

前一次八卡运行（`9a5d308`）不含本次任何被测代码，那批数据对这次**不作数**。

**交付物**：一份 `RESULT.md`（格式见 §5），把**全文复制**发回即可 —— 不需要传任何文件。

**三步流程**（总计约 20–30 分钟，GPU 时间 15–25 分钟）：

```bash
# 1. 拉代码（这份文档和脚本都在里面，不用复制任何东西）
cd /path/to/Relax
git remote add saddss https://github.com/Saddss/Relax.git 2>/dev/null || true
git fetch saddss rloo-review-fix && git checkout -B rloo-review-fix saddss/rloo-review-fix

# 2. 跑四组 + 自动生成报告（约 20 分钟）
MODEL_DIR=/path/to/model DATA_DIR=/path/to/data bash _validation/run_all.sh

# 3. 打开 RESULT.md，补完第 5-7 节，全文复制发回
cat RESULT.md
```

细节、判定标准、陷阱见下文各节。**只想快速跑完的话，上面三条命令就够了。**

______________________________________________________________________

## 0. 同步代码 + 硬闸门

> ⚠️ 闸门不通过就停下。在一份不含被测代码的仓库上"顺利通过"，比不跑更糟。

### 0.1 拉代码

代码已推到 GitHub，直接 fetch 即可，**不需要传任何文件**。

分支：`rloo-review-fix`，仓库 `https://github.com/Saddss/Relax.git`。

```bash
# ↓ 只改这一行
REPO=/path/to/Relax

cd "$REPO"
git remote add saddss https://github.com/Saddss/Relax.git 2>/dev/null || true
git fetch saddss rloo-review-fix
git checkout -B rloo-review-fix saddss/rloo-review-fix
```

若 `git checkout` 报 "local changes would be overwritten"，说明机器上有未提交的改动。**先确认那些改动不是你要保留的东西**，再 `git stash` 或 `git checkout -- .` 清掉，然后重试。不要用 `-f` 硬覆盖之前先看一眼 `git status`。

### 0.2 闸门

```bash
cd "$REPO"

# 不比对固定 SHA（分支可能有后续文档提交）；比对本地是否就是远端分支尖端
git fetch saddss rloo-review-fix
test "$(git rev-parse HEAD)" = "$(git rev-parse saddss/rloo-review-fix)" \
  && echo "✓ 与远端 rloo-review-fix 一致" \
  || { echo "✗ 不是远端分支尖端，执行 git checkout -B rloo-review-fix saddss/rloo-review-fix 后重试"; exit 1; }
git status --porcelain | grep -q . && { echo "✗ 工作区不干净，停止"; exit 1; } || echo "✓ 工作区干净"

# 查符号而不只查 commit：容器里挂载的可能是另一个副本，查符号是查真正被 import 的那份文件
grep -q "def get_cp_local_num_samples" relax/backends/megatron/cp_utils.py \
  && grep -q "def uses_completion_level_reduction" relax/backends/megatron/loss.py \
  && grep -q "effective_grad_num_tokens" relax/backends/megatron/loss.py \
  && grep -q "def validate_rloo_args" relax/utils/training/ppo_utils.py \
  && grep -q 'CONTEXT_PARALLEL_SIZE' examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh \
  && grep -q -- '--clip-grad 0' examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh \
  && echo "✓ 被测代码在位" || { echo "✗ 被测代码缺失，代码是旧的，停止"; exit 1; }
```

容器内自检（用真实 `PYTHONPATH`，验证被 import 的那份）：

```bash
docker run --rm --gpus all -v "$PWD":/w -e PYTHONPATH=/w:/root/Megatron-LM -w /w \
  --entrypoint python3 ghcr.io/redai-infra/relaxrl:latest -c "
from relax.backends.megatron.cp_utils import get_cp_local_num_samples
from relax.backends.megatron.loss import uses_completion_level_reduction
from argparse import Namespace
import torch
m=[torch.ones(4), torch.zeros(4)]
cp1=int(get_cp_local_num_samples([10,10],[4,4],m,dynamic_cp_size=1,dynamic_cp_rank=0))
cp2=[int(get_cp_local_num_samples([10,10],[4,4],m,dynamic_cp_size=2,dynamic_cp_rank=r)) for r in (0,1)]
dt=get_cp_local_num_samples([10,10],[4,4],m,dynamic_cp_size=2,dynamic_cp_rank=0).dtype
print('cp1 =',cp1,'(应为 2)')
print('cp2 per-rank =',cp2,'(应为 [2, 0]，和为 2)')
print('dtype =',dt,'(必须是整数类型)')
print('rloo+per_token =',uses_completion_level_reduction(Namespace(advantage_estimator='rloo',calculate_per_token_loss=True)),'(应为 True)')
print('grpo =',uses_completion_level_reduction(Namespace(advantage_estimator='grpo',calculate_per_token_loss=True)),'(应为 False)')"
```

单卡机器上已跑过这段，输出为 `2` / `[2, 0]` / `torch.int32` / `True` / `False`。多卡机器上应完全一致；不一致说明代码不是这份。

______________________________________________________________________

## 1. 这次要回答什么

按重要性排序，**第 1 条是这次的唯一必答项**，2–4 是同一批运行里顺带就能拿到的。

| # | 问题 | 为什么必须真机验 |
|---|---|---|
| **1** | `get_cp_local_num_samples` 在真实 CP 组下，梯度分母 all-reduce 后是否等于真实 sample 数 | 单测是单进程内把各 rank 结果相加，**没有走真实 all-reduce**；且 Megatron 把这个 normalizer 累加进 `torch.int` 张量（`streaming_schedules.py:106`），非整数会直接 RuntimeError |
| 2 | `train/pg_loss` 的量级是否符合 completion-level 预期 | 归约变了，绝对值必然变化。需要新的基线数据替换 PR 正文 §5 那组旧数 |
| 3 | `rloo_baseline` 在 `CP>1` 下形状是否仍正确、数值是否贴近 `rollout/raw_reward` | 上一版曾用 `response_lengths` 构造导致 shape error（实测分片 740 对全长 1536）；已改为 `ones_like(advantage)`，但只有形状单测 |
| 4 | `pg_clipfrac` 在 `CP>1` 下是否仍恒为 0 | RLOO 不裁剪，非 0 说明走错了分支 |

**明确不在本次范围**：吞吐/MFU 对比、RLOO vs GRPO 的效果结论（单卡上做）、异步模式（启动即拒绝）。

______________________________________________________________________

## 2. 运行矩阵

四组，每组 **3 个 rollout** 就够（要的是"不崩 + 量级对"，不是收敛曲线）。总计约 15–25 分钟。

| 组 | CP | DP | 覆盖的环境变量 | 目的 |
|---|---|---|---|---|
| **A** | 1 | 8 | `ROLLOUT_BATCH_SIZE=8 GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=8` | DP 基准。三个值必须一起改，见 §3 陷阱 1 |
| **B** | 2 | 4 | `CONTEXT_PARALLEL_SIZE=2` | CP=2，主要目标 |
| **C** | 4 | 2 | `CONTEXT_PARALLEL_SIZE=4` | CP=4，分母跨更多 rank |
| **D** | 2 | 4 | `CONTEXT_PARALLEL_SIZE=2 GLOBAL_BATCH_SIZE=16` | **预期启动即失败**：`4 × 8 = 32 ≠ 16`，隐含两个 optimizer step，验证新 guard 生效 |

D 组是负向测试，几秒内就该报错退出，不消耗 GPU 时间。

> **注意**：recipe 不接受命令行透传（脚本内没有 `"$@"`），所有覆盖**必须走环境变量**。本次提交为此把
> `CONTEXT_PARALLEL_SIZE` / `MICRO_BATCH_SIZE` 变成可覆盖的；`--context-parallel-size 2` 这种写法会被静默忽略，
> 跑出来是 CP=1 的结果却以为验了 CP=2。

______________________________________________________________________

## 3. 启动命令

**一条命令跑完四组并生成报告**（脚本已在仓库里，`git checkout` 之后就有）：

```bash
cd "$REPO"
MODEL_DIR=/path/to/shared/model \
DATA_DIR=/path/to/shared/data \
  bash _validation/run_all.sh
```

`MODEL_DIR` 下要有 `Qwen3-0.6B/`，`DATA_DIR` 下要有 `gsm8k/main/train-00000-of-00001.parquet`。默认 8 卡、每组 3 个 rollout；要改用 `NUM_ROLLOUT=5` 或 `CUDA_VISIBLE_DEVICES=...` 覆盖。

`run_all.sh` 会按可见卡数自动设 `RESOURCE='{"actor": [1, N], "rollout": [1, N]}'`。**这一步必须有**：recipe 的默认值是单卡 `[1,1]`，而 Megatron 要求 `world_size % (tp*pp*cp) == 0`，所以 CP=2 在 `[1,1]` 下会直接死在 `validate_args`（`world size (1) is not divisible by total_model_size (2)`），根本进不到被测代码。手动逐组跑时必须自己带上 `RESOURCE`。

**在宿主机跑还是容器里跑**：`collect.sh` 的第 1 节要用 `git` 和 `docker` 采集环境信息，这两个在训练容器里通常都没有。两种做法都可以：

- **在宿主机跑 `run_all.sh`**（推荐）：环境信息完整。注意 recipe 会 `source local.sh`，其中有 `pkill -9 python`，会杀掉宿主机上其它 python 进程 —— 机器上有别的任务时不要这样做。
- **在容器里跑**：安全，但第 1 节的 commit / 镜像 digest / 驱动会显示成 `<...请在宿主机手填>`。按提示在宿主机执行 `git rev-parse HEAD`、`docker images --digests | grep relaxrl`、`nvidia-smi` 补上即可，§2–4 全部从日志读取、不受影响。

脚本做的事：按 A→B→C→D 顺序跑，每组之间 `ray stop --force` + `pkill sglang` 清理残留，日志写到 `/tmp/rloo-*.log`，最后自动执行 `collect.sh` 生成 `./RESULT.md` 并在终端打印跨组一致性判定。

<details>
<summary>若要手动逐组跑（脚本失败时的退路）</summary>

```bash
cd "$REPO"
export MODEL_DIR=/path/to/shared/model DATA_DIR=/path/to/shared/data
export NUM_ROLLOUT=3 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 必须带：默认 [1,1] 无法整除 CP>1 的 total_model_size
export RESOURCE='{"actor": [1, 8], "rollout": [1, 8]}'
R=examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh

ROLLOUT_BATCH_SIZE=8 GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=8 bash $R 2>&1 | tee /tmp/rloo-cp1-dp8.log
ray stop --force; CONTEXT_PARALLEL_SIZE=2 bash $R 2>&1 | tee /tmp/rloo-cp2-dp4.log
ray stop --force; CONTEXT_PARALLEL_SIZE=4 bash $R 2>&1 | tee /tmp/rloo-cp4-dp2.log
ray stop --force; CONTEXT_PARALLEL_SIZE=2 GLOBAL_BATCH_SIZE=16 bash $R 2>&1 | tee /tmp/rloo-negative.log

# 手动跑时 EXIT= 标记不会自动写入，报告里退出码会显示为空 —— 属正常
bash _validation/collect.sh > RESULT.md
```

</details>

### 四个已知陷阱

1. **A 组的 batch 三个值要一起改。** 新 guard 要求 `rollout_batch_size × n_samples_per_prompt == global_batch_size`。`8 × 8 = 64` ✓。若只改 `GLOBAL_BATCH_SIZE=64` 而不改 `ROLLOUT_BATCH_SIZE`，会被 guard 拒绝（`4 × 8 = 32 ≠ 64`）——这**不是 bug，是新校验按预期工作**，但会让你以为跑不起来。
2. **`--calculate-per-token-loss` 必须保持开启**，不要为了"对比"去掉它。recipe 里已开。CP>1 下 Megatron-Bridge 本来就强制要求它（`backends/megatron/arguments.py:96`），且 completion-level 归约依赖它给出的「每样本 token 求和」——关掉它 `uses_completion_level_reduction` 会返回 False，跑的就不是被测路径了。
3. **docker 启动要带 `-w` 和 `--ipc=host`**，不要在宿主机 `source local.sh`（它会 `pkill -9 python`）。
4. **必须确认 `--clip-grad 0` 生效** —— 这条会导致假通过，最关键。

   ```bash
   grep -oE "clip_grad \.+ [0-9.]+" /tmp/rloo-cp2-dp4.log | head -1   # 必须是 0.0
   ```

   completion-level 目标把梯度分母从约 1000 个 token 换成 32 个 sample，`grad_norm` 因此量纲上大约 `response_length` 倍：单卡实测 **248–757**。若沿用 Megatron 默认 `--clip-grad 1.0`，**每一步都会被裁剪**，三组的 `grad_norm` 全被压到 1.0 附近 —— 那样 §4.2 第三条判据（三组是否同量级）就**完全失效**，分母算错也看不出来。本次提交已把 recipe 改成 `--clip-grad 0`；若机器上的 recipe 是旧的，§0 的闸门会先拦住。

______________________________________________________________________

## 4. 判定标准

### 4.1 必须成立（任一不满足即为失败，需要回报）

| 检查 | 判定 |
|---|---|
| A/B/C 三组均完成 3 个 rollout，退出码 0 | 不崩 |
| 日志无 `RuntimeError`、`result type Float can't be cast`、`shape` mismatch | 分母类型与形状正确 |
| 每步 `train/pg_clipfrac == 0.0` | RLOO 未走裁剪分支 |
| 每步 `train/ppo_kl` 有限，无 `nan` / `inf` | 数值健康 |
| `rloo_baseline` 存在且为有限值 | 指标在 CP>1 下没崩 |
| **D 组启动即失败**，报错含 `exactly one optimizer step per rollout` | 新 guard 生效 |

### 4.2 数值合理性（用于替换 PR 正文的数据）

| 量 | 预期 |
|---|---|
| `rloo_baseline` vs `rollout/raw_reward` | 两者应**接近但不相等**。接近是因为组内 advantage 和为零；不相等是因为归约按 token 加权。上一版实测差距 0.002–0.028。**完全相等反而可疑**（说明指标塌成了 reward） |
| `train/pg_loss` 量级 | 会比旧数据大，因为分母从 ~8k tokens 变成 32 samples。量级在 **1e-1 ~ 1e1** 之间属正常；仍应可正可负（unclipped REINFORCE 的固有性质） |
| A/B/C 三组的 `pg_loss` 量级 | 应彼此**同量级**。若 B 或 C 比 A 差了约 `cp_size` 倍，说明分母的 all-reduce 少算或多算了 CP 组 |

注意 `pg_loss` 是**上报值**，仍走 per-token 分母（这是有意保留的，好让它跨 estimator 可比），所以它不会因为本次改动而变化 `cp_size` 倍——真正换了分母的是**梯度**。因此 4.2 第三条的判据要看 `train/grad_norm`：

| 量 | 预期 |
|---|---|
| A/B/C 的 `train/grad_norm` | 三组应同量级。这是梯度分母是否正确 all-reduce 的最直接可观测量；差出约 `cp_size` 倍就是 `get_cp_local_num_samples` 的 rank 归属错了 |
| 与旧数据比 `grad_norm` | 会比旧的（token 分母）**大若干个量级**，因为分母从约 8k tokens 变成 32 samples。这是预期的量纲变化，不是发散；判断稳定性看它在 3 步内是否平稳，而非绝对值 |

> 第三条是本次最关键的判据：它正是"分母 all-reduce 是否正确"的可观测代理。

### 4.3 生成报告

四组跑完后（`run_all.sh` 会自动执行这一步，手动跑时才需要）：

```bash
bash _validation/collect.sh > RESULT.md
```

它会自动产出 §5 报告的第 1-4 节：环境与闸门、A/B/C 逐组逐 step 指标表、D 组 guard 判定、跨组 `grad_norm` 一致性判定。

脚本内三处检查不要改：`cp_size` 防"环境变量没生效、实际跑的是 CP=1"，`clip_grad` 防"被裁剪导致 grad_norm 判据失效"，异常扫描防"报了 PASS 其实中间有 NCCL 错误"。三者任一不符会在报告里直接标 `<-- FAIL`。

______________________________________________________________________

## 5. 交付物：一份 `RESULT.md`

**唯一交付物是 `RESULT.md` 的全文**，直接复制粘贴发回 —— 不需要传任何文件。`collect.sh` 已经生成了第 1–4 节，你只需要补第 5–7 节（人工判断部分）；出问题时再按下文加一个第 8 节。

不要只发结论句、不要只发截图。**没有原始数值的"跑通了"不算交付**。

### 必须包含的 7 个章节

| 节 | 内容 | 来源 |
|---|---|---|
| 1 | 环境与闸门：commit SHA、工作区是否干净、镜像 digest、驱动/CUDA、GPU 型号与数量、四个符号检查是否 PASS | `collect.sh` 自动生成 |
| 2 | 逐组结果：A/B/C 每组一张逐 step 表（9 列，见下）+ 该组的退出码、日志确认的 `cp_size` 与 `clip_grad`、完成 rollout 数、异常扫描输出 | `collect.sh` 自动生成 |
| 3 | 负向组 D：guard 是否触发（YES/NO）+ 报错原文 | `collect.sh` 自动生成 |
| 4 | 跨组一致性：A/B/C 的 `grad_norm` 均值、各步值、组间最大/最小比、PASS/FAIL | `collect.sh` 自动生成 |
| 5 | **逐条结论表**（见下，7 行，每行必须是 PASS / FAIL / N/A + 一句依据） | 你手填 |
| 6 | **遇到的问题**：每个问题写「现象 → 你怎么处理 → 是否影响结论」。**一个都没有就明写「无」**，不要留空 | 你手填 |
| 7 | **一句话总体判定**：`可以合并` / `需要修改（附最可疑的点）` / `无法判断（附原因）` | 你手填 |

### 第 2 节每组的表格列（必须齐全，缺列请写 N/A 并说明）

```
| step | pg_loss | grad_norm | entropy_loss | pg_clipfrac | ppo_kl | rloo_baseline | raw_reward | no_signal |
```

至少 3 行（3 个 rollout）。**数值请保留原始精度，不要四舍五入成两位小数** —— 判断 `cp_size` 倍差异需要精度。

### 第 5 节的结论表（照抄这 7 行填）

```markdown
| # | 检查项 | 结果 | 依据 |
|---|---|---|---|
| 1 | A/B/C 均完成 3 rollout 且退出码 0 | PASS/FAIL | |
| 2 | 日志确认的 cp_size 与各组期望一致（1/2/4） | PASS/FAIL | |
| 3 | 日志确认 clip_grad == 0.0 | PASS/FAIL | |
| 4 | 每步 pg_clipfrac == 0.0 | PASS/FAIL | |
| 5 | 无 RuntimeError / cast 错误 / nan / inf / NCCL 错误 | PASS/FAIL | |
| 6 | rloo_baseline 与 raw_reward 接近但不相等（差 < 0.15 且 ≠ 0） | PASS/FAIL | |
| 7 | A/B/C 的 grad_norm 同量级（组间比 < 3） | PASS/FAIL | |
| 8 | D 组启动即失败且报错含 "exactly one optimizer step per rollout" | PASS/FAIL | |
```

### 出问题时：补一个第 8 节，不要传文件

日志文件传不过来，所以**不要尝试发日志文件**。以下任一成立时，在 `RESULT.md` 末尾加一个第 8 节，把下面这条命令的输出**贴进去**（它只截取相关片段，几十行，可以直接复制）：

- 第 5 节任何一行是 FAIL；
- 第 4 节的组间比 ≥ 3 —— 这是最值得抓到的问题，说明 `get_cp_local_num_samples` 的 rank 归属需要改；
- 任何组的 `grad_norm` 落在单卡基线 196–847 之外一个数量级以上；
- 出现你看不懂的报错。

```bash
bash _validation/excerpt.sh          # 输出直接复制，粘到 RESULT.md 的「## 8. 故障片段」下面
```

它会从四份日志里各截取：最后 40 行、第一处 Traceback 前后 30 行、以及所有含 `Error` / `assert` / `NCCL` / `cast` 的行。够我定位，且不超过几百行。

______________________________________________________________________

## 6. 附：这次到底改了什么（便于判断日志是否合理）

| 文件 | 改动 |
|---|---|
| `relax/backends/megatron/cp_utils.py` | 新增 `get_cp_local_num_samples`：CP rank 0 报全量 sample 数、其余报 0，因此 all-reduce 后恰为真实 sample 数，且**每个 rank 都是整数**（Megatron 把这个 normalizer 累加进 `torch.int` 张量，小数份额会直接 RuntimeError） |
| `relax/backends/megatron/loss.py` | 新增 `uses_completion_level_reduction`；梯度归一化用 `effective_grad_num_tokens`，与**指标**分母 `effective_num_tokens` 分离 |
| `relax/utils/training/ppo_utils.py` | 新增 `validate_rloo_args`（生产与测试共用），含单 step、非零 `--kl-coef`、三个 bypass hook 的校验 |
| `relax/utils/arguments.py` | guard 调用点移到 batch size 派生之后 |

单卡机器上已通过：

- 容器内全量测试 `pytest tests/ --ignore=tests/autoscale`（CI 的调用方式）：**942 passed, 14 skipped, EXIT=0**
- `pre-commit run --all-files`：15 个 hook 全通过
- `get_cp_local_num_samples` 容器内自检：`cp1 = 2`、`cp2 per-rank = [2, 0]`、`dtype = torch.int32`

### 单卡实测基线（1×H100 / Qwen3-0.6B / GSM8K / 6 rollouts / `--clip-grad 0`）

多卡的 A/B/C 三组应与这组**同量级**（DP/CP 只改变数据切分方式，不改变目标函数）：

| step | pg_loss | grad_norm | entropy | clipfrac | rloo_baseline | raw_reward | no_signal |
|---|---|---|---|---|---|---|---|
| 0 | −0.0441 | 483.2 | 0.384 | 0.0 | 0.8491 | 0.8750 | 0.684 |
| 1 | +0.0014 | 196.3 | 0.524 | 0.0 | 0.9815 | 0.9688 | 0.854 |
| 2 | −0.0143 | 847.2 | 0.429 | 0.0 | 0.4606 | 0.4375 | 0.000 |
| 3 | −0.0371 | 656.8 | 0.599 | 0.0 | 0.4802 | 0.5625 | 0.468 |
| 4 | −0.0064 | 422.5 | 0.545 | 0.0 | 0.4339 | 0.4688 | 0.747 |
| 5 | −0.0068 | 398.2 | 0.441 | 0.0 | 0.9649 | 0.9688 | 0.710 |

要点：`clipfrac` 六步恒 0；`grad_norm` **196–847**（这是 completion-level 目标的正常范围，别当成发散——`entropy` 稳定在 0.38–0.60）；`rloo_baseline` 与 `raw_reward` 最大差 0.082 且**从不相等**；`pg_loss` 仍在 1e-2 量级，因为上报指标有意保留 per-token 分母。

`raw_reward` 在 0.44–0.97 之间跳动是采样噪声（每 rollout 只有 4 个 prompt），**不是**趋势，3 个 rollout 的运行不要据此判断学习效果。

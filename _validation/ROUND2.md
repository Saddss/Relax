# 多卡验证 · 第 2 轮 —— 任务 28 RLOO

**这一轮比第 1 轮小得多。** 第 1 轮的核心结论（`grad_norm` 组间比 1.36，分母 all-reduce 正确）**已经成立、不需要重跑**。这轮只为两件事：

1. 第 1 轮之后又补了一条 guard（`--calculate-per-token-loss` 必需），需要确认它不会误伤正常配置；
2. 第 1 轮你是用 `sed` 临时改 `--resource` 才跑通的，工作区不干净；现在 `RESOURCE` 已经是正式参数，想拿一组与代码**逐字节对应**的数据。

如果借不到八卡，**这一轮可以跳过** —— 第 1 轮的证据 + 单卡实测已经够支撑 PR。跳过的话请直接告诉我，我在 PR 里如实写明"CP>1 数据来自含一处临时改动的运行"。

______________________________________________________________________

## 0. 拉代码

```bash
cd /path/to/Relax
git remote add saddss https://github.com/Saddss/Relax.git 2>/dev/null || true
git fetch saddss rloo-review-fix
git checkout -B rloo-review-fix saddss/rloo-review-fix

# 闸门：这轮必须干净，不要再 sed 改任何文件
git status --porcelain | grep -q . && echo "✗ 工作区不干净，停止" || echo "✓ 干净"
test "$(git rev-parse HEAD)" = "$(git rev-parse saddss/rloo-review-fix)" && echo "✓ 是远端尖端" || echo "✗ 不是"
grep -q -- "--resource \"\${RESOURCE}\"" examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh \
  && echo "✓ RESOURCE 已参数化（不再需要 sed）" || echo "✗ 代码是旧的"
grep -q "requires --calculate-per-token-loss" relax/utils/training/ppo_utils.py \
  && echo "✓ 新 guard 在位" || echo "✗ 新 guard 缺失"
```

**四项全绿才往下走。** 第三项就是你上轮 `sed` 改的地方，现在已经是环境变量，这轮不需要改任何文件。

______________________________________________________________________

## 1. 跑什么

**两组**，每组 3 个 rollout，约 10–14 分钟。

| 组 | CP | DP | 目的 |
|---|---|---|---|
| **B** | 2 | 4 | 重跑，这次工作区干净、且 `collect.sh` 的 cp_size 不再误报 |
| **C** | 4 | 2 | 同上 |

A 组（CP=1）不重跑 —— 单卡机器上已用当前代码跑过，数据我这边有。

新加的那条 guard（`--calculate-per-token-loss` 必需）**不需要占用八卡**：它在参数校验阶段就触发，不进入训练。我已在容器里端到端验过 —— 缺 flag 时报 `RLOO requires --calculate-per-token-loss`，`grpo` 不受影响。所以这轮不设负向组。

______________________________________________________________________

## 2. 一条命令

```bash
cd /path/to/Relax
MODEL_DIR=/path/to/shared/model \
DATA_DIR=/path/to/shared/data \
  bash _validation/run_round2.sh
```

脚本会自动设 `RESOURCE='{"actor": [1, 8], "rollout": [1, 8]}'`（按可见卡数推导），跑完 B/C 两组并生成 `./RESULT2.md`。

______________________________________________________________________

## 3. 判定

`collect.sh` 会自动判，你只需确认这 5 行都是 OK/PASS：

| # | 检查 | 期望 |
|---|---|---|
| 1 | B/C 退出码 0，各完成 3 rollout | OK |
| 2 | B 的 cp_size = 2、C 的 cp_size = 4 | **这轮应该显示 OK 了**（上轮的 FAIL 是 grep 误报，已修） |
| 3 | B/C 的 clip_grad = 0.0 | OK |
| 4 | B/C 每步 pg_clipfrac = 0.0、无异常 | OK |
| 5 | B/C 的 `grad_norm` 与上轮同量级（上轮 B=637.8 / C=608.4） | 分母仍然正确 |

`grad_norm` 数值不会与上轮完全一样（采样随机），量级对就行；落在几百这个档位即正常。

### 单卡基线（当前代码，无任何本地改动，`75bf8c4`..`c909b6b`）

用来对比 B/C 是否落在同一档：

| step | pg_loss | grad_norm | entropy | clipfrac | rloo_baseline | raw_reward | no_signal |
|---|---|---|---|---|---|---|---|
| 0 | −0.0265 | 423.0 | 0.382 | 0.0 | 0.8247 | 0.8438 | 0.713 |
| 1 | −0.0374 | 561.0 | 0.581 | 0.0 | 0.9307 | 0.9375 | 0.406 |
| 2 | −0.0273 | 810.9 | 0.447 | 0.0 | 0.3216 | 0.3125 | 0.000 |
| 3 | −0.0729 | 868.7 | 0.605 | 0.0 | 0.4456 | 0.5000 | 0.316 |
| 4 | −0.0215 | 796.8 | 0.545 | 0.0 | 0.4896 | 0.4688 | 0.441 |
| 5 | −0.0180 | 677.2 | 0.445 | 0.0 | 0.8083 | 0.8438 | 0.437 |

`grad_norm` 423–869、`entropy` 0.382–0.605、`clipfrac` 恒 0、`baseline` 与 `raw_reward` 最大差 0.054 且从不相等、`ppo_kl` 全有限。日志确认 `calculate_per_token_loss=True`、`clip_grad=0.0`，即走的正是被测的 completion-level 路径。

______________________________________________________________________

## 4. 交付

打开 `RESULT2.md`，在末尾补三行，然后**全文复制**发回：

```markdown
## 5. 结论

| # | 检查项 | 结果 |
|---|---|---|
| 1 | B/C 退出码 0 且各 3 rollout | PASS/FAIL |
| 2 | cp_size 显示 2 / 4（上轮误报已修） | PASS/FAIL |
| 3 | clip_grad = 0.0 | PASS/FAIL |
| 4 | pg_clipfrac 全 0、异常扫描为空 | PASS/FAIL |
| 5 | grad_norm 与上轮同量级 | PASS/FAIL |

## 6. 工作区是否干净
本轮是否修改过任何文件： 是/否（若是，写明改了什么）

## 7. 一句话
可以合并 / 需要修改（附最可疑的点）
```

第 6 节请务必如实填 —— 这轮的意义之一就是拿一组"没有任何本地改动"的数据。

出问题时：`bash _validation/excerpt.sh`，输出直接复制粘贴。

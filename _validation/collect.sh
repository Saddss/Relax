#!/bin/bash
# 用法: bash collect.sh > RESULT.md
set -u
declare -A LOGS=( [A]=/tmp/rloo-cp1-dp8.log [B]=/tmp/rloo-cp2-dp4.log [C]=/tmp/rloo-cp4-dp2.log )
declare -A EXPECT_CP=( [A]=1 [B]=2 [C]=4 )
declare -A EXPECT_DP=( [A]=8 [B]=4 [C]=2 )

echo "# 多卡验证结果 —— 任务 28 RLOO"
echo
echo '## 1. 环境与闸门'
echo
echo '```'
echo "commit          : $(git rev-parse HEAD)"
_remote_tip=""
for _r in saddss origin; do
  _t=$(git rev-parse "$_r/rloo-review-fix" 2>/dev/null) && { _remote_tip=$_t; break; }
done
echo "分支            : $(git rev-parse --abbrev-ref HEAD)  $([ -n "$_remote_tip" ] && { [ "$(git rev-parse HEAD)" = "$_remote_tip" ] && echo '= 远端 rloo-review-fix，OK' || echo '<-- 与远端 rloo-review-fix 不一致'; } || echo '(未找到远端 rloo-review-fix，跳过比对)')"
echo "工作区干净      : $(git status --porcelain | grep -q . && echo NO || echo YES)"
echo "镜像 digest     : $(docker images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep relaxrl | head -1)"
echo "驱动 / CUDA     : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) / $(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | head -1)"
echo "GPU             : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x $(nvidia-smi -L | wc -l)"
echo '符号检查        :' \
  "$(grep -q 'def get_cp_local_num_samples' relax/backends/megatron/cp_utils.py && \
     grep -q 'def uses_completion_level_reduction' relax/backends/megatron/loss.py && \
     grep -q 'effective_grad_num_tokens' relax/backends/megatron/loss.py && \
     grep -q 'def validate_rloo_args' relax/utils/training/ppo_utils.py && echo PASS || echo FAIL)"
echo '```'
echo
echo '## 2. 逐组结果'
for g in A B C; do
  f=${LOGS[$g]}
  echo
  echo "### 组 $g （期望 CP=${EXPECT_CP[$g]}, DP=${EXPECT_DP[$g]}）"
  echo
  if [ ! -f "$f" ]; then echo '**日志不存在 —— 该组未运行**'; continue; fi
  echo '```'
  echo "日志文件          : $f"
  exit_code=$(grep -oE 'EXIT=[0-9]+' "$f" | tail -1 | cut -d= -f2)
  got_cp=$(grep -oE 'context_parallel_size \.+ [0-9]+' "$f" | tail -1 | grep -oE '[0-9]+$')
  got_clip=$(grep -oE 'clip_grad \.+ [0-9.]+' "$f" | tail -1 | grep -oE '[0-9.]+$')
  echo "退出码            : ${exit_code:-<无 EXIT= 标记，运行未结束>}  $([ "${exit_code:-x}" = 0 ] && echo OK || echo '<-- FAIL')"
  echo "日志确认 cp_size  : ${got_cp:-?}  (期望 ${EXPECT_CP[$g]})  $([ "${got_cp:-x}" = "${EXPECT_CP[$g]}" ] && echo OK || echo '<-- FAIL: 环境变量没生效，跑的不是目标 CP')"
  echo "日志确认 clip_grad: ${got_clip:-?}  (必须 0.0)  $([ "${got_clip:-x}" = "0.0" ] && echo OK || echo '<-- FAIL: 被裁剪，grad_norm 判据失效')"
  echo "完成 rollout 数   : $(grep -c 'Rollout fully completed' "$f")"
  echo '```'
  echo
  echo '| step | pg_loss | grad_norm | entropy_loss | pg_clipfrac | ppo_kl | rloo_baseline | raw_reward | no_signal |'
  echo '|---|---|---|---|---|---|---|---|---|'
  python3 - "$f" <<'PY'
import re,sys
log=open(sys.argv[1],errors='ignore').read()
def g(k,pre='train'): return re.findall(r"'%s/%s': (-?[0-9.eE+-]+)"%(pre,k),log)
cols=[g('pg_loss'),g('grad_norm'),g('entropy_loss'),g('pg_clipfrac'),g('ppo_kl'),
      g('rloo_baseline'),g('raw_reward','rollout'),g('rloo_no_signal_fraction')]
n=min((len(c) for c in cols), default=0)
for i in range(n):
    print('| %d | '%i + ' | '.join(c[i] for c in cols) + ' |')
if n==0: print('| (无 train 步指标，训练未进入 loss 计算) |'+' |'*8)
PY
  echo
  echo '异常扫描（应为空）:'
  echo '```'
  # 过滤掉启动时回显的参数表和环境变量（它们含 NCCL / nan / inf 字样但不是错误）
  grep -nE "RuntimeError|can't be cast|Traceback|shape mismatch|CUDA error|NCCL error|NCCL WARN|\bnan\b|\binf\b" "$f" \
    | grep -viE "avail_mem|Warning|warn|check_for_nan|inference_batch|NVSHMEM|NCCL_|nccl version|^\s*[0-9]+:\s+\"|\.+ (True|False|None|-?[0-9])" \
    | head -8
  echo '(以上为空则通过)'
  echo '```'
done
echo
echo '## 3. 负向组 D'
echo
echo '```'
if grep -q "exactly one optimizer step per rollout" /tmp/rloo-negative.log 2>/dev/null; then
  echo "guard 触发 : YES"
  echo "报错原文   :"
  grep -A2 "exactly one optimizer step per rollout" /tmp/rloo-negative.log | head -6
else
  echo "guard 触发 : NO  <-- 这是问题，请把 /tmp/rloo-negative.log 一起发回"
fi
echo '```'
echo
echo '## 4. 跨组一致性（本次核心判据）'
echo
echo '```'
python3 - "${LOGS[A]}" "${LOGS[B]}" "${LOGS[C]}" <<'PY'
import re,sys,statistics
names=['A(CP1)','B(CP2)','C(CP4)']
means={}
for name,path in zip(names,sys.argv[1:]):
    try: log=open(path,errors='ignore').read()
    except OSError: print(f'{name}: 日志缺失'); continue
    v=[float(x) for x in re.findall(r"'train/grad_norm': (-?[0-9.eE+-]+)",log)]
    if not v: print(f'{name}: 无 grad_norm'); continue
    means[name]=statistics.mean(v)
    print(f'{name}: grad_norm 均值 ={means[name]:.1f}  各步={[round(x,1) for x in v]}')
if len(means)>1:
    lo,hi=min(means.values()),max(means.values())
    print(f'\n组间最大/最小比 = {hi/lo:.2f}')
    print('判定: ' + ('PASS（同量级）' if hi/lo < 3 else 'FAIL（疑似分母按 cp_size 算错，请附三份完整日志）'))
PY
echo '```'

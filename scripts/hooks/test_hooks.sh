#!/usr/bin/env bash
# 实验流水账 hook 的回归测试。直接跑: ./scripts/hooks/test_hooks.sh
#
# 这些 hook 平时是静默的——判断错了不会报错，只会不该提醒时提醒、该提醒时沉默。
# 所以每条判断规则都在这里钉一个用例，改正则之前先看它们还过不过。
set -uo pipefail

HOOKS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pass=0 fail=0

ok() { pass=$((pass + 1)); printf '  ok    %s\n' "$1"; }
no() { fail=$((fail + 1)); printf '  FAIL  %s\n' "$1"; }
check() { [[ "$2" == "$3" ]] && ok "$1" || no "$1 (期望 $3，实际 $2)"; }

# 造一个临时仓库，把 hook 拷进去，用给定的 JOURNAL 正文。
scratch() {
    local dir; dir="$(mktemp -d)"
    mkdir -p "$dir/scripts/hooks" "$dir/docs/experiments" "$dir/.claude"
    cp "$HOOKS"/*.sh "$dir/scripts/hooks/"
    { echo '<!-- ENTRIES -->'; cat; } > "$dir/docs/experiments/JOURNAL.md"
    printf '%s' "$dir"
}

# --- exp_journal_check.sh: 什么算一次启动 ---------------------------------
echo "exp_journal_check: 启动识别"
DIR="$(scratch </dev/null)"
probe() {
    printf '{"tool_input":{"command":%s},"tool_response":{"exit_code":%s}}' \
        "$(jq -Rn --arg c "$1" '$c')" "${2:-0}" \
        | "$DIR/scripts/hooks/exp_journal_check.sh" \
        | grep -q additionalContext && echo nag || echo silent
}

check "训练启动器"            "$(probe 'bash scripts/run_train_seed0.sh')" nag
check "带环境变量的 ./ 启动"  "$(probe 'CKPT=x OUT=y ./scripts/dump_rspr_uncertainty.sh')" nag
check "torchrun"              "$(probe 'torchrun --nproc_per_node=4 main_task_retrieval.py --do_train')" nag
check "离线分析脚本"          "$(probe 'python scripts/conformal_coverage_probe.py --run a=b.npz')" nag
check "绝对路径的解释器"      "$(probe '/home/x/envs/tvr/bin/python scripts/probe_rspr_uncertainty.py')" nag
check "复合命令里的启动"      "$(probe 'git log --oneline -3 && bash scripts/eval.sh')" nag
check "前面挂了个 nvidia-smi" "$(probe 'nvidia-smi && bash scripts/run_train_seed0.sh')" nag

check "只是在看脚本"          "$(probe 'cat scripts/final_test_from_ckpt.sh')" silent
check "在 grep 脚本名"        "$(probe 'grep -n rspr_mode main_task_retrieval.py')" silent
check "脚本名只是字符串"      "$(probe 'echo "run scripts/eval.sh later" > notes.txt')" silent
check "--help 不是实验"       "$(probe 'python main_task_retrieval.py --help')" silent
check "失败的命令不提醒"      "$(probe 'bash scripts/run_train_seed0.sh' 1)" silent
check "无关命令"              "$(probe 'pytest tests/')" silent

# --- exp_journal_check.sh: 什么算已登记 -----------------------------------
echo "exp_journal_check: 登记状态"
mk() { DIR="$(scratch <<<"$1")"; }

mk '## [2026-07-31 02:00] step-002 在跑
- status: RUNNING
- run_dir: ckpts/bbb
- 证伪条件: spread 没降到 10pt 以下就算死'
check "最后一条 RUNNING 且有证伪条件 -> 闭嘴" "$(probe 'bash scripts/run_train_seed0.sh')" silent

mk '## [2026-07-31 02:00] step-002 在跑但没写判据
- status: RUNNING
- run_dir: ckpts/bbb'
check "RUNNING 但缺证伪条件 -> 提醒"          "$(probe 'bash scripts/run_train_seed0.sh')" nag

mk '## [2026-07-31 02:00] step-002 判据是空的
- status: RUNNING
- run_dir: ckpts/bbb
- 证伪条件:'
check "证伪条件写了空值 -> 提醒"              "$(probe 'bash scripts/run_train_seed0.sh')" nag

mk '## [2026-07-31 01:00] step-001 陈旧的没收尾
- status: RUNNING
- run_dir: ckpts/aaa
- 证伪条件: x

## [2026-07-31 02:00] step-002 已收尾
- status: DONE
- run_dir: ckpts/bbb
- 证伪条件: y'
check "陈旧 RUNNING 不再永久关掉提醒"         "$(probe 'bash scripts/run_train_seed0.sh')" nag

# --- _journal_lib.sh: 解析 -------------------------------------------------
echo "_journal_lib: 解析"
DIR="$(scratch <<'EOF'
## [2026-07-31 01:00] step-001 翻过状态的
- status: RUNNING
- run_dir: ckpts/aaa
- 证伪条件: x
- status: DONE

## [2026-07-31 02:00] step-002 训练类，走默认完成信号
- status: RUNNING
- run_dir: ckpts/bbb
- 证伪条件: y

## [2026-07-31 03:00] step-003 分析类，显式完成信号，没写判据
- status: RUNNING
- run_dir: -
- 完成信号: .scratch/unc/parity_a4.npz
EOF
)"
cd "$DIR"; source scripts/hooks/_journal_lib.sh
check "RUNNING 条目数（最后一次 status 生效）" "$(journal_running | wc -l)" 2
check "默认完成信号"     "$(journal_running | sed -n 1p | cut -f1)" "ckpts/bbb/final_test.json"
check "显式完成信号"     "$(journal_running | sed -n 2p | cut -f1)" ".scratch/unc/parity_a4.npz"
check "证伪条件有无"     "$(journal_running | cut -f3 | tr '\n' ,)" "yes,no,"
check "最后一条的状态"   "$(journal_last_entry | cut -f1)" "RUNNING"

# --- exp_run_done.sh: 完成检测 ---------------------------------------------
echo "exp_run_done: 完成检测"
run_done() { "$DIR/scripts/hooks/exp_run_done.sh" >/dev/null 2>&1; echo $?; }
check "都没落地 -> 不唤醒" "$(run_done)" 0

mkdir -p "$DIR/.scratch/unc" && : > "$DIR/.scratch/unc/parity_a4.npz"
check "分析类产物落地 -> 唤醒" "$(run_done)" 2
check "同一信号只唤醒一次"     "$(run_done)" 0

mkdir -p "$DIR/ckpts/bbb" && echo '{}' > "$DIR/ckpts/bbb/final_test.json"
check "训练类产物落地 -> 唤醒" "$(run_done)" 2
check "两个都记过了"           "$(run_done)" 0

# --- exp_context.sh: 注入 ---------------------------------------------------
echo "exp_context: 注入"
printf 'x\n%.0s' {1..45} > "$DIR/docs/experiments/STATUS.md"
out="$("$DIR/scripts/hooks/exp_context.sh" SessionStart)"
check "输出是合法 JSON"     "$(jq -e . <<<"$out" >/dev/null 2>&1 && echo yes)" yes
ctx="$(jq -r '.hookSpecificOutput.additionalContext' <<<"$out")"
check "带了当前时间"        "$(grep -cq '当前时间' <<<"$ctx" && echo yes)" yes
check "STATUS 超长有提示"   "$(grep -cq '超过 40 行上限' <<<"$ctx" && echo yes)" yes
check "缺判据的 RUNNING 有提示" "$(grep -cq '没写 证伪条件' <<<"$ctx" && echo yes)" yes

# 多行 HTML 注释不能漏进正文
printf '# S\n<!--\n内部备注不该出现\n-->\n正文\n' > "$DIR/docs/experiments/STATUS.md"
ctx="$("$DIR/scripts/hooks/exp_context.sh" SessionStart | jq -r '.hookSpecificOutput.additionalContext')"
check "多行注释被剥掉"      "$(grep -cq '内部备注' <<<"$ctx" && echo leaked || echo clean)" clean
check "正文还在"            "$(grep -cq '正文' <<<"$ctx" && echo yes)" yes

echo
echo "通过 $pass，失败 $fail"
[[ $fail -eq 0 ]]

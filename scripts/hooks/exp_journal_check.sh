#!/usr/bin/env bash
# PostToolUse(Bash) hook: 启动了实验但流水账里没有对应条目时提醒补写。
#
# 不拦截。训练是后台跑的，启动后几秒内补写条目，同样满足"证伪条件写在看到结果之前"。
# 便宜的离线分析脚本不满足这一点——它们几十秒就出结果，提醒到达时数字已经在屏幕上了。
# 但那类实验可以反复重跑、最容易事后合理化，所以宁可迟到也要提醒：迟到的证伪条件
# 至少留下了"当时打算怎么判"，缺席的连这个都没有。
set -uo pipefail

# shellcheck source=_journal_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_journal_lib.sh"

input="$(cat)"
cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // ""' 2>/dev/null)"
[[ -n "$cmd" ]] || exit 0

# 失败的命令不提醒：flag 打错秒退、路径不存在之类的，没产生任何需要记录的实验。
# 各家 harness 放退出码的键名不一样，取到哪个算哪个；一个都取不到就按成功处理。
status="$(printf '%s' "$input" | jq -r '
    [.tool_response.exit_code?, .tool_response.exitCode?, .tool_response.returncode?]
    | map(select(type == "number")) | first // empty' 2>/dev/null)"
[[ -n "$status" && "$status" != "0" ]] && exit 0

# 训练/评估的启动脚本，加上现在这一阶段真正在跑的离线分析。原先只认前者，
# 结果 conformal gate、σ 分层、噪声扫描这些实验一条流水账都不会被要求写——
# 而它们恰恰最需要：几十秒就能重跑一次，事后合理化的空间比训练大得多。
LAUNCH_RE='main_task_retrieval|torchrun'
LAUNCH_RE+='|run_train[^[:space:]]*\.sh|run_[^[:space:]]*seed[0-9][^[:space:]]*\.sh'
LAUNCH_RE+='|(^|[/[:space:]])(eval|train_msvd|final_test_from_ckpt|dump_rspr_uncertainty)\.sh'
LAUNCH_RE+='|conformal_coverage_probe|probe_rspr_uncertainty|fire_corrected_metrics'
grep -Eq "$LAUNCH_RE" <<<"$cmd" || exit 0

# 命令里得真有个执行动作，不能只是提到了脚本名。原先的排除规则锚定整条命令的第一个
# 词，所以 `git log && bash scripts/eval.sh` 被当成 git 放过；反过来，把脚本名当字符串
# 传给别的东西（定义 shell 函数、写 grep 模式）会被当成启动。改成正面匹配执行动词：
# python / torchrun / bash / sh 出现在命令位置，或者 ./ 直接执行。
EXEC_RE='(^|[;&|]|&&|\|\|)[[:space:]]*([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*'
EXEC_RE+='([^[:space:]]*/)?(python[0-9.]*|torchrun|bash|sh|nohup|srun)[[:space:]]'
EXEC_RE+='|(^|[;&|[:space:]])(\./|[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+\./)[^[:space:]]+\.sh'
grep -Eq "$EXEC_RE" <<<"$cmd" || exit 0

# --help / --version 不是实验。
grep -Eq '(^|[[:space:]])--(help|version)([[:space:]]|$)' <<<"$cmd" && exit 0

# 已经登记过了吗？看最后一条条目，不是"存在任意 RUNNING"。规矩是一步一条、跑之前
# 先写，所以刚启动时最新一条就该是刚写的那条。原先按"存在任意 RUNNING"判断，一条
# 忘了收尾的陈旧条目会把提醒永久关掉，同时手上有两个 run 时第二个也不提醒。
last="$(journal_last_entry)"
last_status="$(cut -f1 <<<"$last")"
last_fals="$(cut -f2 <<<"$last")"

if [[ "$last_status" == "RUNNING" && "$last_fals" == "yes" ]]; then
    exit 0
fi

if [[ "$last_status" == "RUNNING" && "$last_fals" == "no" ]]; then
    jq -Rs '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: .}}' <<'MSG'
流水账最后一条是 status: RUNNING，但它没有写 `- 证伪条件:`（或者写了空的）。

现在补上，趁结果还没出来。一句话，说清楚看到什么就算这个假设死了。
没有它，这条记录只能证明"跑过了"，没法让任何人——包括上下文全丢的你自己——
判断这个 run 算成功还是失败。
MSG
    exit 0
fi

jq -Rs '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: .}}' <<'MSG'
刚刚启动了一个实验，但 docs/experiments/JOURNAL.md 的最后一条不是 status: RUNNING，
也就是没有为这次启动登记。

现在就追加一条（趁结果还没出来）：

## [YYYY-MM-DD HH:MM] step-NNN 一句话标题
- status: RUNNING
- run_dir: <本次的输出目录，没有就写 ->
- 完成信号: <哪个文件出现就算跑完；训练类可省略，默认 run_dir/final_test.json>
- 假设: 一句话，可证伪的那种
- 证伪条件: 看到什么就算这个假设死了
- 命令: 刚跑的那一行

时间戳用 `date '+%Y-%m-%d %H:%M'` 取，不要凭印象填。

证伪条件必须现在写。等结果出来再补，它就只是在描述已经发生的事，
挡不住事后合理化，也没法让一个上下文全丢的后继者判断这个 run 算成功还是失败。
MSG

exit 0

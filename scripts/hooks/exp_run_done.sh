#!/usr/bin/env bash
# Stop hook (asyncRewake): run 的完成信号已经落地，但流水账那条还挂着 RUNNING ——
# exit 2 把模型唤回来补结论。
#
# 训练在 agent 的回合之外结束，没有任何工具调用会通知它。文件系统是唯一可靠的完成信号。
# 完成信号取自条目里的 `- 完成信号:`，没写就退回 `<run_dir>/final_test.json`；原先写死
# 后者，于是产出 .npz 的离线分析永远不会被认成"跑完了"。
#
# 每个信号只唤醒一次（记在 .claude/.exp_nagged），否则模型不照做就会变成死循环。
# 只唤醒一次意味着不照做这条就永远沉默——兜底在 SessionStart：exp_context.sh 每次
# 开场都会把所有 RUNNING 条目重新列出来，所以欠账不会真的丢。
set -uo pipefail

# shellcheck source=_journal_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_journal_lib.sh"

NAG_FILE="$EXP_ROOT/.claude/.exp_nagged"
touch "$NAG_FILE" 2>/dev/null || exit 0

finished=()
while IFS=$'\t' read -r signal run_dir _fals title; do
    [[ -n "$signal" ]] || continue
    [[ "$signal" = /* ]] && abs="$signal" || abs="$EXP_ROOT/$signal"
    [[ -e "$abs" ]] || continue
    grep -Fxq "$signal" "$NAG_FILE" && continue
    finished+=("${run_dir:-$signal}"$'\t'"$title"$'\t'"$signal")
done < <(journal_running)

[[ ${#finished[@]} -gt 0 ]] || exit 0

{
    echo "以下 run 的完成信号已经落地，但 docs/experiments/JOURNAL.md 里对应条目还是 status: RUNNING："
    echo
    for row in "${finished[@]}"; do
        IFS=$'\t' read -r run_dir title signal <<<"$row"
        echo "  - $run_dir"
        echo "    ${title#\#\# }"
        echo "    完成信号: $signal"
    done
    echo
    echo "现在去读结果，然后："
    echo "  1. 在该条目下追加 结果:/结论:/下一步:，再追加一行 '- status: DONE'；"
    echo "  2. 对照条目里写好的**证伪条件**判定，不要重新发明一个更宽松的标准；"
    echo "  3. 如果这一步让方向变了，覆盖更新 docs/experiments/STATUS.md（≤40 行）；"
    echo "  4. 成段的论证和表格才写进 docs/experiments/baseline-gap-diagnosis.md。"
} >&2

# 去重按完成信号记，和上面的判断口径一致。
printf '%s\n' "${finished[@]##*$'\t'}" >> "$NAG_FILE"

exit 2

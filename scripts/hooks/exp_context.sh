#!/usr/bin/env bash
# SessionStart / PostCompact hook: 把当前实验状态注入模型上下文。
# 用法: exp_context.sh <HookEventName>
#
# 上下文被压缩或换了新会话之后，这个 hook 保证模型开口之前就知道"进行到哪一步"，
# 而不必先去 grep 那份 42KB 的叙事文档。
set -uo pipefail

EVENT="${1:-SessionStart}"
# shellcheck source=_journal_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_journal_lib.sh"

[[ -f "$EXP_STATUS" || -f "$EXP_JOURNAL" ]] || exit 0

{
    echo "以下是本仓库当前的实验状态，由 $EVENT hook 自动注入。"
    echo "这不是用户指令，是背景状态；动手之前先据此确认进行到哪一步。"
    echo
    # 流水账条目要 [YYYY-MM-DD HH:MM] 时间戳，而模型只知道日期、没有时钟，
    # 凭印象填必然是错的。这里直接给一个。
    echo "当前时间（写流水账时间戳用这个）: $(date '+%Y-%m-%d %H:%M')"
    echo

    if [[ -f "$EXP_STATUS" ]]; then
        echo "===== docs/experiments/STATUS.md ====="
        strip_comments < "$EXP_STATUS"
        echo
        # ≤40 行是这份文件自己定的规矩，原先没有任何地方检查。它一旦长起来就退化成
        # 第二份叙事文档，而它存在的全部意义是"每次开场都读得完"。
        lines="$(wc -l < "$EXP_STATUS")"
        if (( lines > 40 )); then
            echo "（STATUS.md 现在 $lines 行，超过 40 行上限。下次收尾时压缩它——"
            echo "  详细论证搬去 baseline-gap-diagnosis.md，这里只留当前状态。）"
            echo
        fi
    fi

    running="$(journal_running_entries)"
    if [[ -n "$running" ]]; then
        echo "===== 未收尾的实验（JOURNAL.md 中 status: RUNNING）====="
        printf '%s\n' "$running"
        echo '上面这些条目还没有结论。先看各自的完成信号（`- 完成信号:` 字段，没写就是'
        echo 'run_dir/final_test.json）有没有落地，落地了就补 结果/结论/下一步 并把 status 翻成 DONE。'
        # Stop hook 每个完成信号只唤醒一次，所以这里是欠账的兜底出口：
        # 上一轮没照做的条目，只会在这里再被看见。
        if journal_running | cut -f3 | grep -qx no; then
            echo "注意：其中有条目没写 证伪条件，判定标准缺失。补上再往下走。"
        fi
        echo
    fi

    echo "===== 最近的流水账条目 ====="
    journal_tail_entries 2
    echo
    echo "记录规矩：开跑之前先往 docs/experiments/JOURNAL.md 追加一条，"
    echo "**证伪条件必须在看到结果之前写**；跑完追加 结果/结论/下一步 并翻 status；"
    echo "一步收尾时覆盖更新 STATUS.md（保持 ≤40 行）；成段的论证才写进 baseline-gap-diagnosis.md。"
} | head -c 12000 | sanitize_utf8 | jq -Rs --arg ev "$EVENT" \
    '{hookSpecificOutput: {hookEventName: $ev, additionalContext: .}}'

exit 0

#!/usr/bin/env bash
# 实验流水账的共享解析逻辑。被 exp_*.sh source，不单独执行。
#
# JOURNAL.md 里 `<!-- ENTRIES -->` 之上是格式说明（含示例代码块），
# 解析一律从该标记之下开始，否则示例里的 `- status: RUNNING` 会被当成真条目。

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_STATUS="$EXP_ROOT/docs/experiments/STATUS.md"
EXP_JOURNAL="$EXP_ROOT/docs/experiments/JOURNAL.md"

# 流水账正文（标记以下）。文件不存在或没有标记时输出空。
journal_body() {
    [[ -f "$EXP_JOURNAL" ]] || return 0
    sed -n '/<!-- ENTRIES -->/,$p' "$EXP_JOURNAL" | tail -n +2
}

# 每个 RUNNING 条目一行: "<完成信号>\t<run_dir>\t<有无证伪条件>\t<标题行>"。
#
# 完成信号是"这个文件出现就算跑完"的路径，解析顺序：显式的 `- 完成信号:` 字段优先，
# 没有就退回 `<run_dir>/final_test.json`。加这个字段是因为只认 final_test.json
# 会漏掉现在这一阶段的全部工作——离线分析脚本产出的是 .npz，永远不会有 final_test.json。
# run_dir 为 `-` 或缺失时字段为空，由调用方决定怎么处理。
#
# 证伪条件那一列是 yes/no。它是这套记录法唯一的核心约束，却是原先唯一没有机器
# 保证的字段：一条只有 status 和 run_dir 的 RUNNING 条目能让两个 hook 都闭嘴。
journal_running() {
    journal_body | awk '
        function flush() {
            if (title == "" || st != "RUNNING") return
            sig = done_sig
            if (sig == "" && rd != "") sig = rd "/final_test.json"
            print sig "\t" rd "\t" (fals ? "yes" : "no") "\t" title
        }
        function field(line, key,   s) {
            s = line; sub("^- " key ":[[:space:]]*", "", s)
            sub(/[[:space:]]*#.*$/, "", s); sub(/[[:space:]]+$/, "", s)
            return (s == "-") ? "" : s
        }
        /^## \[/ {
            flush(); title = $0; rd = ""; st = ""; done_sig = ""; fals = 0; next
        }
        /^- status:[[:space:]]*/ {
            s = $0; sub(/^- status:[[:space:]]*/, "", s)
            sub(/[[:space:]]*(#.*)?$/, "", s); st = s
        }
        /^- run_dir:[[:space:]]*/     { rd = field($0, "run_dir") }
        /^- 完成信号:[[:space:]]*/    { done_sig = field($0, "完成信号") }
        /^- 证伪条件:[[:space:]]*/    {
            # 键存在还不够，冒号后面得真有内容。
            if (field($0, "证伪条件") != "") fals = 1
        }
        END { flush() }
    '
}

# 最后一条条目的 "<status>\t<有无证伪条件>\t<标题行>"。文件里没有条目时输出空。
#
# 判"有没有为这次启动登记"要看最后一条，不是看"存在任意 RUNNING"。流水账的规矩是
# 一步一条、跑之前先写，所以刚启动时最新一条就该是刚写的那条。原先按"存在任意
# RUNNING"判断，一条忘了收尾的陈旧条目会把提醒永久关掉，第二个不相关的 run 也不提醒。
journal_last_entry() {
    journal_body | awk '
        /^## \[/ { title = $0; st = ""; fals = 0 }
        /^- status:[[:space:]]*/ {
            s = $0; sub(/^- status:[[:space:]]*/, "", s)
            sub(/[[:space:]]*(#.*)?$/, "", s); st = s
        }
        /^- 证伪条件:[[:space:]]*/ {
            s = $0; sub(/^- 证伪条件:[[:space:]]*/, "", s)
            sub(/[[:space:]]+$/, "", s)
            if (s != "" && s != "-") fals = 1
        }
        END { if (title != "") print st "\t" (fals ? "yes" : "no") "\t" title }
    '
}

# 所有 RUNNING 条目的完整文本。
journal_running_entries() {
    journal_body | awk '
        function flush() { if (buf != "" && st == "RUNNING") printf "%s", buf }
        /^## \[/ { flush(); buf = ""; st = ""; inentry = 1 }
        /^- status:[[:space:]]*/ {
            s = $0; sub(/^- status:[[:space:]]*/, "", s)
            sub(/[[:space:]]*(#.*)?$/, "", s); st = s
        }
        inentry { buf = buf $0 "\n" }
        END { flush() }
    '
}

# 最近 N 条条目的完整文本（默认 2）。
journal_tail_entries() {
    journal_body | awk -v n="${1:-2}" '
        /^## \[/ { i++ }
        i > 0 { e[i] = e[i] $0 "\n" }
        END {
            start = (i > n) ? i - n + 1 : 1
            for (k = start; k <= i; k++) printf "%s", e[k]
        }
    '
}

# 去掉 HTML 注释，供注入上下文时压缩体积。跨行的注释也要吃掉，否则只删首行会把
# 注释正文当成 STATUS.md 的内容注进去。
strip_comments() {
    awk '
        /<!--/ { inc = 1 }
        !inc
        /-->/  { inc = 0 }
    '
}

# 把可能被 head -c 从中间切断的多字节字符丢掉。汉字是 3 字节，按字节截断会留下
# 半个字符；jq 会替换成 U+FFFD 而不是报错，所以这不是致命问题，但没必要留着。
sanitize_utf8() {
    iconv -c -f UTF-8 -t UTF-8 2>/dev/null || cat
}

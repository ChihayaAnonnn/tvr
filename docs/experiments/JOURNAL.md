# 实验流水账 (append-only)

一步一条，**跑之前先写**，跑完追加结论。只追加，不改写历史条目
（除了把 `status:` 从 RUNNING 翻成 DONE/ABANDONED，以及补 `结论:`/`下一步:`）。

叙事和论证归 `baseline-gap-diagnosis.md`；当前状态归 `STATUS.md`；
这里只记"第 N 步做了什么、期望什么、实际什么"。

## 条目格式

`status:` / `run_dir:` / `完成信号:` / `证伪条件:` 是**机器字段**，hook 靠它们判断，
必须顶格以 `- <键>:` 开头。其余字段给人看。

```markdown
## [YYYY-MM-DD HH:MM] step-NNN 一句话标题
- status: RUNNING
- run_dir: ckpts/xxx          # 没有产出目录就写 -
- 完成信号: ckpts/xxx/final_test.json   # 哪个文件出现就算跑完
- 假设: 一句话，可证伪的那种
- 证伪条件: 看到什么就算这个假设死了（**必须在跑之前写**）
- 命令: 复现用的那一行
```

时间戳用 `date '+%Y-%m-%d %H:%M'` 取。SessionStart 注入里也带了当前时间，
别凭印象填——模型没有时钟。

`完成信号:` 省略时默认 `<run_dir>/final_test.json`，所以训练类不用写。**离线分析
必须写**（`.scratch/xxx.npz` 之类），否则 Stop hook 认不出它跑完了。`run_dir` 是 `-`
且没写完成信号的条目，只能靠人手动收尾。

`证伪条件:` 不是给人看的散文，是 hook 会检查存在性的字段：写了空值等于没写，
PostToolUse 和 SessionStart 都会揪出来。

跑完在同一条目下追加：

```markdown
- status: DONE
- 结果: 数字，带单位和对照
- 结论: 假设成立/不成立，为什么
- 下一步: 或者写"无，回 STATUS.md 重新定向"
```

`status:` 的取值只有三个：`RUNNING` / `DONE` / `ABANDONED`。
同一条目里 `status:` 出现多次时以**最后一次**为准，所以翻状态就是再追加一行。

---

<!-- ENTRIES -->
<!-- hook 只解析这一行以下的内容，上面的格式说明不会被当成条目。新条目追加到文件末尾。 -->

## [2026-07-31 03:16] step-000 建立流水账本身
- status: DONE
- run_dir: -
- 假设: 实验记录断裂的原因不是没人写，而是"在飞状态"和"自动回灌"两处没有机器保证——
  manifest 记了跑什么配置、从不记为什么跑；叙事文档要等结论落地才补写，
  恰好在实验进行中那段窗口是空的；而 42KB 的叙事文档没法每次开场重读。
- 证伪条件: 若 SessionStart/PostCompact 注入后，新会话仍需要人工指路才知道进行到哪一步，
  说明 STATUS.md 的内容选得不对，不是机制的问题。
- 命令: -
- 结果: STATUS.md / JOURNAL.md / 四个 hook 就位。
- 结论: 待用。这一条本身就是格式样例。
- 下一步: 下一步实验按 STATUS.md 的 (c)：文本编码器在 |Rel| 上 fine-tune 的歧义预测器。

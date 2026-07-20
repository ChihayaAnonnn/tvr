# MSR-VTT 单文件后台训练入口设计

## 1. 目标

将当前两级启动链路：

```text
run_train_msrvtt_bg.sh -> train_msrvtt.sh -> torchrun -> main_task_retrieval.py
```

合并为单一入口 `run_train_msrvtt_bg.sh`，同时保持以下用户行为：

- 训练进程由 `setsid` 在后台启动；
- 标准输出和错误输出统一写入按日期组织的日志文件；
- 启动终端立即执行 `tail -n 50 -F`；
- `Ctrl-C` 只停止日志跟随，不终止后台训练；
- 可选的 `TRAIN_PID_FILE` 继续记录后台会话 PID；
- 环境变量和尾随 CLI 参数继续传递给最终的 Python 训练入口。

合并后删除 `train_msrvtt.sh`。用户继续使用原命令：

```bash
./run_train_msrvtt_bg.sh [main_task_retrieval.py CLI 参数...]
```

## 2. 范围与非目标

本次只改变 shell 启动结构，不改变训练算法或训练默认值。

必须保留当前工作树中的有效训练行为，包括：

- trusted split 使用 `msrvtt_trusted_v1_seed0.json`；
- 训练命令包含 `--run_final_test`；
- A800、hygiene、batch、GPU、CLIP checkpointing 和参数覆盖保护规则；
- `"$@"` 对 `main_task_retrieval.py` CLI 的完整透传。

本次不负责：

- 新增或调整 RSPR 环境变量；该工作仍属于 RSPR 实施计划的 Task 8；
- 修改 Python 训练、数据加载或 checkpoint 逻辑；
- 修改训练超参数、GPU 拓扑或日志目录约定；
- 启动真实训练作为自动化测试。

## 3. 方案选择

采用“同一脚本、控制器与 worker 双模式”方案。

没有内部标记时，脚本运行控制器模式；控制器通过一个保留的内部环境变量重新启动同一脚本。子进程检测到该标记后进入 worker 模式，不再创建新的后台进程，从而避免递归。

内部环境变量固定命名为：

```text
RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER=1
```

它不是公共配置接口，不写入用户文档，也不转成 Python CLI 参数。

## 4. 单文件结构

`run_train_msrvtt_bg.sh` 按职责拆为两个 shell 函数：

```text
main
├── run_controller
│   ├── 生成 RUN_DATE / RUN_TIME / RUN_TAG / RUN_ID
│   ├── 校验 RUN_TAG
│   ├── 创建日志目录和日志文件
│   ├── setsid 启动当前脚本的 worker 模式
│   ├── 记录 TRAIN_PID_FILE
│   └── tail -n 50 -F 日志
└── run_worker
    ├── 设置线程和缓存环境
    ├── 解析数据、split、输出目录和训练配置
    ├── 构建 trusted split
    ├── 执行参数、GPU 与 hygiene 校验
    ├── 组装可选 CLIP 参数
    └── torchrun main_task_retrieval.py "$@"
```

脚本末尾只负责模式分发：

```bash
if [[ "${RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER:-0}" == "1" ]]; then
    run_worker "$@"
else
    run_controller "$@"
fi
```

## 5. 控制器模式

控制器保留现有日期、标签、日志和 PID 行为。后台命令使用当前脚本的绝对路径：

```bash
setsid env \
    RUN_ID="${RUN_ID}" \
    RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER=1 \
    bash "${ROOT_DIR}/run_train_msrvtt_bg.sh" "$@" \
    >"${LOG_FILE}" 2>&1 &
```

关键约束：

- `"$@"` 必须逐项保留，不能拼成字符串或通过 `eval` 执行；
- 日志重定向覆盖 worker 中的 split 构建、校验、`torchrun` 和 Python 输出；
- `TRAIN_PID=$!` 与当前行为一致，记录 `setsid` 启动的后台会话进程；
- `tail` 保持在控制器前台，不能把训练 PID 绑定到 `tail` 生命周期；
- 不增加自动 kill、重启或重试行为。

## 6. Worker 模式

Worker 内联当前 `train_msrvtt.sh` 的完整工作树内容，不从已提交的旧版本重新复制。这样可保留用户尚未提交的 `seed0` manifest 和 `--run_final_test` 修改。

进入 Worker 后立即取消导出内部模式变量，避免它泄漏到 split 构建进程、`torchrun` 或 Python 训练进程：

```bash
unset RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER
```

Worker 继续：

1. 设置 `TORCH_WARN_ONCE` 和 BLAS/OpenMP 线程变量；
2. 解析数据路径、缓存、split manifest 和生成目录；
3. 调用 `scripts/build_msrvtt_trusted_split.py`；
4. 解析训练、CLIP、GPU、batch 和 hygiene 配置；
5. 拒绝受保护选项被尾随参数覆盖；
6. 根据 `CUDA_VISIBLE_DEVICES` 计算或校验 `NPROC`；
7. 使用 `torchrun` 启动 `main_task_retrieval.py`；
8. 将调用者提供的 CLI 参数放在命令最后继续覆盖非受保护参数。

Worker 不负责日志文件创建和 `tail`，也不得再次调用控制器。

## 7. 错误处理与退出语义

- 控制器中的 `RUN_TAG` 错误在启动前直接输出到终端并返回状态 2；
- Worker 中的 split、配置、GPU 或训练错误写入本次训练日志；
- Worker 失败后日志跟随仍可显示错误，行为与当前两脚本链路一致；
- `set -euo pipefail` 在控制器和 worker 中都生效；
- 不使用 `eval`，避免参数中的空格或特殊字符被重新解释；
- 不吞掉 `torchrun` 的退出状态。

## 8. 兼容与引用迁移

- 删除 `train_msrvtt.sh`，`run_train_msrvtt_bg.sh` 成为唯一受支持的 MSR-VTT 训练入口；
- `run_train_bg.sh` 保持现有弃用跳转，仍指向 `run_train_msrvtt_bg.sh`；
- 更新仍在使用的脚本说明、诊断文档字符串和当前 RSPR Task 8 计划，使其指向新入口；
- 历史归档计划和已经完成的设计文档不做批量改写；
- 实现提交不得包含其他工作树修改。

## 9. 测试设计

新增针对统一脚本的自动化测试，所有外部命令都通过临时 `PATH` 中的伪命令替代，不启动真实训练。

### 9.1 语法与引用

- `bash -n run_train_msrvtt_bg.sh run_train_bg.sh` 通过；
- 活动代码和当前操作文档不再调用 `train_msrvtt.sh`；
- `train_msrvtt.sh` 不再存在。

### 9.2 控制器测试

用伪 `setsid` 和伪 `tail` 运行控制器，验证：

- 后台命令重新调用 `run_train_msrvtt_bg.sh`；
- 设置内部 worker 标记和相同 `RUN_ID`；
- 原始 CLI 参数逐项保留；
- 日志路径包含指定的 `RUN_DATE/RUN_TIME/RUN_TAG`；
- `TRAIN_PID_FILE` 被写入；
- `tail` 接收 `-n 50 -F` 和正确日志路径。

### 9.3 Worker 测试

以内部 worker 标记运行脚本，并使用伪 `python3` 与伪 `torchrun`，验证：

- Worker 不调用 `setsid` 或 `tail`，不会递归；
- trusted split 构建命令参数保持不变；
- `torchrun` 接收正确 GPU 进程数、Python 入口、固定训练参数和尾随 RSPR CLI；
- 当前 `seed0` manifest 与 `--run_final_test` 行为被保留；
- 非法 GPU、hygiene 或受保护参数仍以状态 2 失败。

## 10. 验收条件

实现只有在以下条件全部满足时才完成：

1. 项目只保留一个受支持的 MSR-VTT 训练实现文件；
2. 原有 `run_train_msrvtt_bg.sh` 调用方式不变；
3. 后台训练、日志重定向、PID 文件和前台 `tail` 行为均有测试；
4. 数据准备、参数校验、GPU 配置和 `torchrun` 参数与合并前等价；
5. 尾随 CLI 参数能无损到达 `main_task_retrieval.py`；
6. 用户当前对 `train_msrvtt.sh` 的工作树修改已迁移，没有丢失；
7. shell 语法检查和相关项目测试通过；
8. 没有启动真实训练、修改模型功能或提前实现 Task 8 的 RSPR shell 配置。

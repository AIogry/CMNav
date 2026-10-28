# CMNav

## 离散双层 GCML 阶段检查点（2026-10-28）

本阶段在 `proto/pipeline` 保存离散经验图上的 CML/GCML、landmark 双层规划、
读出拟合、GPU 实验流程和分档噪声对照实现。高低层均使用离散节点与有向边，
模型内部的连续潜向量想象不等于连续几何路径规划；新的空间低层尚未接入。

- 双层结构与训练：`src/cmnav/data/landmark_hierarchy.py`、
  `src/cmnav/methods/gcml.py`、`src/cmnav/workflows/landmark_hierarchy.py`。
- 冻结评测与噪声分析：`src/cmnav/evaluation/hierarchical_gcml.py`、
  `src/cmnav/analysis/hierarchy_noise.py`、`src/cmnav/reporting/hierarchy_noise.py`。
- 统一入口：`cmnav experiment`；阶段协议与调用配置位于 `configs/experiments/`，
  用法见 [scripts/README.md](scripts/README.md)。历史运行引用需要按本机资产位置配置。
- 此检查点保存源码、配置和现有测试文件；归档操作没有重新训练、评测或运行测试。
  不能将提交完成视为新的实验验证。
- 按现有忽略规则，`docs/`、场景与轨迹数据、`outputs/` 中的模型和结果、
  `.env/` 与 `.env-gpu/` 保留在本地。Git 检查点是代码归档，不能独立恢复完整实验资产。

CMNav 是面向 CML / GCML 视觉导航研究的代码框架。现有环境检查、探索采集、轨迹离线读取和记忆快照之外，`proto/pipeline` 已加入官方 PointNav 参考采集、经验图、CML 学习与导航对照的原型流程。

**当前 CML / GCML 原型使用真值位置、NavMesh 和官方局部控制器，属于有特权信息的工程诊断。** RGB-D 已记录，但尚未用于模型状态编码；视觉状态关联、视觉运动估计和 PPO 尚未接入。成功执行流程不等于模型已有有效视觉导航表现，具体结果需绑定运行目录与代码版本。协议、命令和信息权限见 [图导航诊断说明](docs/图导航诊断说明.md)。

MP3D 已获得使用资格，并已取得官方下载脚本。当前采用官方单场景包中的 `17DRP5sb8fy` 与对应 PointNav train 分片，使用 `configs/mp3d_smoke.yaml` 运行；场景、任务、下载缓存均存放在 `CMNav/data`。另提供 `configs/habitat_test_smoke.yaml`，用于公开 Habitat 测试场景的渲染检查。

**第一阶段历史验证：**独立环境已安装；初版骨架通过 40 项测试。公开 Castle 场景与 MP3D `17DRP5sb8fy` 均已在 GPU 1 完成 40 步运行，各保存 41 次 RGB-D 观测。MP3D 已准备 81,967 个单场景训练任务；当时只运行其中一个任务用于检查环境。完整记录与结果入口见 [第一阶段记录](docs/before10-22/milestone_01.md)。

**第二阶段验证完成：**全量 209 项测试通过；真实 MP3D 采集 2 条 × 64 动作，每次共 130 次观测；同一清单重复运行的动作、RGB-D 和审计位姿完全一致。采集、保存与离线读取的小规模闭环已跑通，见 [第二阶段验证记录](docs/before10-22/milestone_02_validation.md)。

**首批开发数据已采集：**单场景 10 条 × 500 动作，共 5,010 次 RGB-D 观测，约 1.7 GiB；完整性与离线读取检查通过。前进碰撞较多，质量诊断和数据入口见 [10×500 开发采集记录](docs/before10-22/mp3d_pilot_10x500.md)。

## 当前图导航诊断入口

默认配置 `configs/mp3d_oracle_graph.yaml` 冻结 20 个学习任务和 5 个开发评测任务，每任务最多 500 个实际动作。任务都来自同一个 MP3D 场景的 train 分片，不能作为官方测试集或场景泛化成绩。

```bash
cd /home/eai/Navigation/CMNav
# 已有默认任务清单；直接复用，不重复生成覆盖。
.env/bin/python -m cmnav diagnose-graph-navigation \
  --config configs/mp3d_oracle_graph.yaml \
  --plan data/navigation_plans/196ef31279f361ce1046978db4cb1fad5d4328ec1a3fadf79c00daf2cc619ecb.json
```

图方法 `dijkstra`、`cml` 和 `random_graph` 使用相同的经验图、局部控制器、任务与预算；`oracle_follower` 使用完整 NavMesh，单独作为参考。当前诊断在每个任务中使用 Frozen Map，正式研究的任务内 Online Map + 任务结束丢弃协议继续保留。新清单生成、`--reference-run` 复用参考经验和产物说明见 [图导航诊断说明](docs/图导航诊断说明.md)。

**2026-10-22 开发诊断：**20 个参考任务全部成功，保存 1,232 个动作和 1,252 次 RGB-D 观测；5 个开发任务中，官方跟随器成功 5 个、Dijkstra 成功 1 个、CML 与随机图策略均为 0。只有 1 个任务的端点被经验图支持且图可达，CML 在该任务上仍未成功。当前结论是工程链路已接通，覆盖和 CML 导航行为仍需改进；具体运行与限制见 [实际运行记录](docs/图导航诊断说明.md#7-2026-10-22-实际运行记录)。该轮未新增或运行单元测试。

**2026-10-23 CML 训练实现更新：**默认 hidden state 为 1000 维，后端改为 PyTorch float32；按作者实现整条轨迹批量更新、每轮打乱轨迹顺序。默认在已观测有向图上生成 200 条随机游走，每条最多 32 个状态（31 次转移），训练 10 轮；这属于图上重放，不是新增 RGB-D 采集。新模型为 `cmnav.cml.v2`，旧 v1 模型只支持读取推理。实现调整阶段只做静态与依赖检查；上述历史成绩仍属于 NumPy、128 维版本。

**2026-10-23 首轮 PyTorch 实验已完成：**复用同一经验图与 5 个任务，1000 维、10 轮训练，实际消费 37,830 次转移。官方跟随器成功 5/5、Dijkstra 1/5、CML 和随机图 0/5。在唯一受支持任务上，CML 19 动作后进入图中死端，Dijkstra 64 动作成功；CML 全程没有碰撞或前进卡住。详细条件、关键分叉评分和路线图见 [实验报告](/home/eai/Navigation/CMNav/outputs/oracle_graph_navigation/20261023T085250Z_c0e69165/experiment_report.md)。未因结果调整参数重跑，也未运行单元测试。

## 结构与兼容

代码按 [代码结构与命名规范](docs/代码结构与命名规范.md) 归入职责子包。入口调用流程，流程组合数据、记忆、方法与环境；数据与算法不反向导入运行器。配置检查、文件指纹和来源记录提供公共接口，源码指纹覆盖完整 `cmnav` 包。

旧 `prototype` 命令暂时保留弃用别名，执行同一个诊断实现。配置路径改用 `configs/mp3d_oracle_graph.yaml`；默认新输出为 `outputs/oracle_graph_navigation/`。内部 Python 导入已同步迁移，不保留第二套旧模块。历史数据、模型和 `outputs/pointnav_prototype/` 不搬迁，协议与 schema 标识不因目录变化而改变。

**2026-10-23 结构迁移验证：**219 项测试通过，35 个模块导入不加载 Habitat；旧配置解析、微型 CML 更新及选边与迁移前一致，历史模型、图、快照和轨迹可以读取。GPU 1 上重放任务 `81735`，Dijkstra 仍以 64 个动作成功，动作序列和最终指标与历史结果相同。未重新训练 MP3D CML；验证范围与日志见 [迁移验证说明](docs/图导航诊断说明.md#9-2026-10-23-结构迁移验证)。

这份结构迁移验证发生在 PyTorch 算法调整之前，不作为当前训练实现的验证结论。当前模型及图导航运行需安装 `learning` 依赖；配置、轨迹文件和采集工具仍可独立使用。

## 配置驱动的实验入口

训练、冻结评测、读出拟合、分析和报告通过同一能力注册表调用。后续更换维度、轮数、种子或条件时，新增配置及运行目录，复用包内实现：

```bash
.env/bin/python -m cmnav experiment --list
.env/bin/python -m cmnav experiment --config configs/experiments/train_gcml_3000.yaml --describe
.env/bin/python -m cmnav experiment --operation evaluate-gcml --profile condition -- --help
```

`--describe` 只展示解析结果。实际 GPU 训练使用 `.env-gpu/bin/python`，移除 `--describe`，并在配置中指定新的输出目录。两份 3000/5000 维配置是复用入口的示例，本轮整理未执行它们。

2026-10-26 已将 21 个训练、分析和报告脚本改为兼容入口，业务分别归入 `workflows/`、`methods/`、`evaluation/`、`analysis/`、`reporting/` 和 `runtime/`。旧文件名暂留，数据准备工具保留独立职责。统一入口保存调用配置及源码快照；协议差异仍由明确的 profile 和契约校验，不因入口统一而合并。

用法和旧命令映射见 [scripts 使用说明](scripts/README.md)，迁移范围与检查限制见 [scripts 管理与能力归并记录](docs/scripts管理与能力归并记录.md)。

## 探索采集与记忆

- [采集与离线读取](docs/before10-22/trajectory_collection.md)：`plan-collection` 固定起点清单，`collect` 按清单执行，`trajectory-info` 读取完整轨迹。开发配置为每场景 10 条、每条 500 个动作。
- [记忆接口](docs/before10-22/memory.md)：不可变初始快照、任务工作副本、任务结束丢弃，以及显式的跨任务更新提交。
- [已确认协议](docs/before10-22/trajectory_protocol_v1.md)：首轮主设置为固定初始快照 + 任务内 Online Map + 任务结束丢弃 + 冻结共享神经网络参数；保留 Frozen Map 消融。

当前记忆接口管理状态与生命周期，CML 原型已使用冻结快照接入导航。实际视觉自定位、视觉跨轨迹关联、任务内 CML 地图更新仍需实现；共享神经网络的冻结由后续视觉模型接入负责。

## 开始使用

项目目录是 `/home/eai/Navigation/CMNav`。依赖安装见 [环境搭建](docs/before10-22/setup.md)，数据目录说明见 [数据准备](data/README.md)。现有 Conda 环境按路径创建，位置为 `/home/eai/Navigation/CMNav/.env`，使用 `conda activate /home/eai/Navigation/CMNav/.env` 激活。它不是一个已注册名称为 `cmnav` 的环境。目标版本为 Python 3.9、Habitat-Sim 0.3.3 headless、Habitat-Lab 0.3.3、NumPy 1.26.4，以及新增的 PyTorch 2.8.0。

在已配置 Habitat 的环境中，安装 CML 学习依赖：

```bash
cd /home/eai/Navigation/CMNav
.env/bin/python -m pip install -e '.[learning]'
```

本次采用 CPU 版 PyTorch，CML 默认 `cml_device: cpu`。重建同类环境时，可先安装 CPU wheel，再执行上面的 editable 安装：

```bash
.env/bin/python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
```

代码支持在图导航 YAML 中独立选择 `cuda:N`，使用前需将 PyTorch 更换为匹配机器和驱动的 CUDA 构建。CML 与 Habitat 的渲染 GPU 分别管理，CPU 版 PyTorch 不改变 Habitat 原有的 GPU 1 渲染设置。需要运行完整开发测试时还需安装 `dev` 依赖。

GPU 机制实验使用项目内的 `.env-gpu`，这是继承 `.env` 依赖的 Python venv，单独安装 CUDA 版 PyTorch 2.8.0。它不是 Conda 环境名称，运行时直接指定解释器即可。重建命令如下，CUDA 12.6 wheel 来自 [PyTorch 官方版本安装页](https://pytorch.org/get-started/previous-versions/)：

```bash
.env/bin/python -m venv --system-site-packages .env-gpu
.env-gpu/bin/python -m pip install 'torch==2.8.0+cu126' --index-url https://download.pytorch.org/whl/cu126
```

`scripts/train_gcml_checkpoints.py` 按运行目录的 `protocol.json` 在 CUDA 上训练并发布每 20 轮 checkpoint；`scripts/evaluate_gcml_mechanisms.py` 执行单个冻结条件；`scripts/run_gcml_gpu_mechanisms.py --run <运行目录>` 调度 GPU1 训练与 GPU0/1 评估；完成后用 `scripts/report_gcml_gpu_mechanisms.py --run <运行目录>` 汇总。GCML 张量运算要求 CUDA，设备不可用时直接失败。CPU 仍负责仿真控制、日志和文件存储，Dijkstra 使用常规图搜索。驱动被沙箱隔离时，需在允许访问 NVIDIA 驱动的宿主执行环境启动。

```bash
cd /home/eai/Navigation/CMNav
.env/bin/python scripts/prepare_mp3d_data.py
.env/bin/python -m cmnav doctor --config configs/mp3d_smoke.yaml --output outputs/doctor_mp3d.json
.env/bin/python -m cmnav run --config configs/mp3d_smoke.yaml
```

`doctor` 检查包信息、episode 文件和其引用的场景资源；缺少依赖或数据时退出码为 `2`。它不会初始化 GPU、EGL 或 NavMesh。静态检查通过后，仍需实际 `run` 验证场景加载和渲染。

也可先下载公开测试数据并执行渲染检查：

```bash
.env/bin/python scripts/download_test_data.py
.env/bin/python -m cmnav doctor --config configs/habitat_test_smoke.yaml --output outputs/doctor_habitat_test.json
.env/bin/python -m cmnav run --config configs/habitat_test_smoke.yaml
```

默认配置使用 GPU `1`。如果机器只提供一张 GPU，可在运行时指定 `--gpu 0`。其他常用覆盖项为 `--episodes`、`--max-steps`、`--seed` 和 `--output-dir`：

```bash
.env/bin/python -m cmnav run --config configs/mp3d_smoke.yaml --gpu 0 --episodes 1 --max-steps 20
```

## 环境检查命令的输入与协议

- 观测包含 `rgb`、`depth` 和 `pointgoal`。默认 RGB-D 分辨率为 256 × 256，深度未归一化，单位为米。
- `pointgoal` 是目标在 episode 起始位姿坐标系中的固定三维笛卡尔向量，采用 Habitat 的坐标约定：x 向右、y 向上、−z 向前。它不会随着机器人运动更新。已确定最终保留 Fixed、Estimated 与 Oracle 三种目标接口；当前只实现 Fixed 的环境检查，另两种尚未实现，设计见 [轨迹协议](docs/before10-22/trajectory_protocol_v1.md)。
- 动作为 `MOVE_FORWARD`、`TURN_LEFT`、`TURN_RIGHT`、`STOP`。默认前进 0.25 米、转向 30 度，每个 episode 最多 40 步。
- 随机策略默认以 0.6 的概率前进，剩余概率均分给左右转向，并在达到步数上限的最后一步主动 `STOP`。它没有学习目标到达或停止判断。
- 距离、Success、SPL、SoftSPL、碰撞与步数供记录和评测使用。环境真值指标不作为策略输入。`execution_status: completed` 表示运行流程结束，不表示导航成功；导航结果需查看 `metrics.success`。
- 当前 episode 迭代器在数据耗尽后循环；日志保留真实场景和 episode 标识。后续科学评测需要另外固定独立 episode 集合，不能把重复运行当成新增任务。

以上适用于 `run` 环境检查。新增 `collect` 仅给策略 RGB-D，不提供目标；起点清单使用一次，外部预算结束标为截断，不插入 STOP。真值诊断单独保存在 `audit/`。

## 文件职责

```text
CMNav/
├── configs/                           # 参数和协议配置
│   ├── mp3d_smoke.yaml
│   ├── mp3d_collect.yaml
│   ├── mp3d_oracle_graph.yaml
│   └── habitat_test_smoke.yaml
├── src/cmnav/
│   ├── __init__.py, __main__.py, cli.py
│   ├── configuration/                 # 校验、环境、采集和图导航配置
│   ├── data/                          # 轨迹读写、任务清单与图随机游走
│   ├── envs/                          # Habitat 适配、特权导航和资源检查
│   ├── policies/                      # 随机检查策略与深度探索策略
│   ├── memory/                        # 快照生命周期与真值辅助经验图
│   ├── methods/cml.py                 # PyTorch 局部学习和图边评分
│   ├── workflows/                     # 环境检查、探索/参考采集与诊断编排
│   ├── evaluation/oracle_graph_navigation.py # 图导航任务执行和汇总
│   ├── analysis/                      # 保存记录的读取、诊断与比较
│   ├── reporting/                     # 轨迹图、比较图和报告模板
│   └── runtime/                       # 来源记录、文件工具、episode 记录
├── data/                              # 实际数据资源，与源码 data/ 区分
├── scripts/                           # 独立准备工具和薄兼容入口
├── tests/                             # 按真实职责检查迁移后的模块
├── docs/图导航诊断说明.md               # 当前权限、用法与历史结果
├── PROTOTYPE.md                        # 旧说明链接的迁移入口
├── WORKTREES.md                        # proto/pipeline 与 proto/data 协作
└── outputs/                           # 新旧运行各自保留原始身份
```

YAML 中的 `project_root` 相对 **YAML 所在目录** 解析；`episodes_path`、`scenes_dir`、可选的 `scene_dataset_config` 和 `output_dir` 再相对 `project_root` 解析。MP3D 显式加载官方场景配置，以记录资源关联与坐标约定。默认配置中的 `project_root: ..` 指向 CMNav。`episodes_path` 支持 `{split}` 占位符。命令行 `--config` 相对当前工作目录解析。

## 环境检查运行产物

每次运行使用独立的 UTC 时间戳与随机后缀目录，例如 `outputs/<run_id>/`：

| 文件 | 内容 |
| --- | --- |
| `config.yaml` | 应用覆盖项并展开路径后的 CMNav 配置 |
| `runtime.json` | Python、平台、包版本和解释器路径 |
| `habitat_config.json` | 环境建立后导出的 Habitat 配置 |
| `preflight.json` | 成功运行后保存的静态检查报告 |
| `summary.json` | 流程状态、episode 汇总、平均指标和运行时间 |
| `episode_0000/episode.json` | episode、场景标识与策略随机种子 |
| `episode_0000/steps.jsonl` | 每次观测对应的前一步动作、目标向量、指标和帧路径 |
| `episode_0000/frames/*.rgb.png` | RGB 图像 |
| `episode_0000/frames/*.depth.npy` | float32 深度数据，单位为米 |
| `episode_0000/frames/*.depth.png` | 为查看而缩放至 8 位的深度图 |
| `episode_0000/preview.gif` | RGB 预览，最多采样 200 帧 |
| `episode_0000/summary.json` | episode 的动作数、终止原因与最终指标 |
| `error.txt` | 运行阶段异常的堆栈，仅失败时生成 |

`steps.jsonl` 的第 `0` 条记录是 `reset()` 后的初始观测，`action_from_previous_observation` 为 `null`；第 `t` 条记录中的动作把第 `t−1` 次观测带到第 `t` 次观测。默认每步保存图像，`run.save_every` 可降低保存频率，终止帧仍会保存。

后续分析应读取 `.depth.npy`；`.depth.png` 和 GIF 仅用于查看。GIF 的播放速度与保存频率可配置，不代表模拟器的真实运动时间。

## 下一步

Landmark 层次原型通过 `cmnav experiment` 的 `hierarchy/build`、`hierarchy/train`、`hierarchy/evaluate`、`hierarchy/report` 四个能力运行，配置位于 `configs/experiments/*landmark_hierarchy*`。从同一份有序图重放构建宏关系和定向接入/接出表，冻结细层模型，独立训练高层 GCML；当前仅在抽象图上做四组对照。协议、实现及运行记录见 [Landmark 层次 GCML 原型](docs/10-26/Landmark层次GCML原型.md)。

当前主线已转向 GCML，保留 CML 代码和历史结果。GCML 的 PyTorch Q/V/W/G 学习、连续潜状态想象、完整候选选择、逐边重规划及独立图审计已接入；入口为 `cmnav diagnose-gcml-navigation`，配置为 `configs/mp3d_gcml.yaml`。使用独立冻结的 5 个图支持任务，与 Dijkstra、同 checkpoint 单步读出及经验图辅助可用性版本比较。具体设定、来源差异、命令和结果见 [GCML 首轮实现与实验](docs/10-23/GCML首轮实现与实验.md)。

教师分阶段诊断已接入 `cmnav diagnose-gcml-subgoals --config configs/mp3d_gcml_subgoals.yaml`。同一 checkpoint 在相同五任务上的经验图辅助版本，从最终目标直达的 0/5 提升到每 4 条边提供正确子目标的 5/5；学得 G 的版本仍为 0/5。Dijkstra 提供的子目标属于额外信息，不能将结果视为自主层次规划。设定、全部 40 次任务结果及路线图见 [GCML 分阶段目标实验](docs/10-24/GCML分阶段目标实验.md)。

后续根据候选覆盖、可用性预测和实际执行诊断推进，再接入视觉状态表示、运动估计及在线地图更新。正式实验还需要更多场景、冻结采集预算及数据版本，并明确训练、调参和独立评测任务划分。当前 GCML 是官方 notebook 派生的 PointNav 适配，不能称为完整复现论文或正式视觉导航结果。

`proto/data` 独立开发高质量研究数据，并通过版本化数据与读取约定交付；`proto/pipeline` 负责当前算法和导航流程，协作方式见 [WORKTREES.md](WORKTREES.md)。严格组合评测需要专门的数据与任务设计，当前同场景开发结果不能替代它。代码保留持续记忆接口；首轮正式实验仍按 [已确认协议](docs/before10-22/trajectory_protocol_v1.md) 采用任务内更新、任务间恢复快照。

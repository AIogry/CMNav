# CMNav

CMNav 是面向 CML / GCML 视觉导航研究的代码框架。现有环境检查可在 Habitat 中运行 RGB-D PointNav episode；第二阶段已加入固定起点的探索采集、完整轨迹离线读取和场景记忆快照接口。

**当前尚未实现 CML、GCML、视觉状态关联、认知地图学习或导航训练。** 随机策略仅用于检查环境和数据流；其成功率不能作为研究结论。

MP3D 已获得使用资格，并已取得官方下载脚本。当前采用官方单场景包中的 `17DRP5sb8fy` 与对应 PointNav train 分片，使用 `configs/mp3d_smoke.yaml` 运行；场景、任务、下载缓存均存放在 `CMNav/data`。另提供 `configs/habitat_test_smoke.yaml`，用于公开 Habitat 测试场景的渲染检查。

**第一阶段历史验证：**独立环境已安装；初版骨架通过 40 项测试。公开 Castle 场景与 MP3D `17DRP5sb8fy` 均已在 GPU 1 完成 40 步运行，各保存 41 次 RGB-D 观测。MP3D 已准备 81,967 个单场景训练任务；当时只运行其中一个任务用于检查环境。完整记录与结果入口见 [第一阶段记录](docs/milestone_01.md)。

**第二阶段验证完成：**全量 209 项测试通过；真实 MP3D 采集 2 条 × 64 动作，每次共 130 次观测；同一清单重复运行的动作、RGB-D 和审计位姿完全一致。采集、保存与离线读取的小规模闭环已跑通，见 [第二阶段验证记录](docs/milestone_02_validation.md)。

**首批开发数据已采集：**单场景 10 条 × 500 动作，共 5,010 次 RGB-D 观测，约 1.7 GiB；完整性与离线读取检查通过。前进碰撞较多，质量诊断和数据入口见 [10×500 开发采集记录](docs/mp3d_pilot_10x500.md)。

## 探索采集与记忆

- [采集与离线读取](docs/trajectory_collection.md)：`plan-collection` 固定起点清单，`collect` 按清单执行，`trajectory-info` 读取完整轨迹。开发配置为每场景 10 条、每条 500 个动作。
- [记忆接口](docs/memory.md)：不可变初始快照、任务工作副本、任务结束丢弃，以及显式的跨任务更新提交。
- [已确认协议](docs/trajectory_protocol_v1.md)：首轮主设置为固定初始快照 + 任务内 Online Map + 任务结束丢弃 + 冻结共享神经网络参数；保留 Frozen Map 消融。

当前记忆接口管理状态与生命周期，CML 的地图更新规则、自定位、跨轨迹关联和导航接入仍需实现；共享神经网络的冻结由后续模型接入负责。

## 开始使用

项目目录是 `/home/eai/Navigation/CMNav`。依赖安装见 [环境搭建](docs/setup.md)，数据目录说明见 [数据准备](data/README.md)。现有 Conda 环境按路径创建，位置为 `/home/eai/Navigation/CMNav/.env`，使用 `conda activate /home/eai/Navigation/CMNav/.env` 激活。它不是一个已注册名称为 `cmnav` 的环境。目标版本为 Python 3.9、Habitat-Sim 0.3.3 headless、Habitat-Lab 0.3.3、NumPy 1.26.4。

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
- `pointgoal` 是目标在 episode 起始位姿坐标系中的固定三维笛卡尔向量，采用 Habitat 的坐标约定：x 向右、y 向上、−z 向前。它不会随着机器人运动更新。已确定最终保留 Fixed、Estimated 与 Oracle 三种目标接口；当前只实现 Fixed 的环境检查，另两种尚未实现，设计见 [轨迹协议](docs/trajectory_protocol_v1.md)。
- 动作为 `MOVE_FORWARD`、`TURN_LEFT`、`TURN_RIGHT`、`STOP`。默认前进 0.25 米、转向 30 度，每个 episode 最多 40 步。
- 随机策略默认以 0.6 的概率前进，剩余概率均分给左右转向，并在达到步数上限的最后一步主动 `STOP`。它没有学习目标到达或停止判断。
- 距离、Success、SPL、SoftSPL、碰撞与步数供记录和评测使用。环境真值指标不作为策略输入。`execution_status: completed` 表示运行流程结束，不表示导航成功；导航结果需查看 `metrics.success`。
- 当前 episode 迭代器在数据耗尽后循环；日志保留真实场景和 episode 标识。后续科学评测需要另外固定独立 episode 集合，不能把重复运行当成新增任务。

以上适用于 `run` 环境检查。新增 `collect` 仅给策略 RGB-D，不提供目标；起点清单使用一次，外部预算结束标为截断，不插入 STOP。真值诊断单独保存在 `audit/`。

## 文件职责

```text
CMNav/
├── configs/                  # 实验配置
│   ├── mp3d_smoke.yaml
│   ├── mp3d_collect.yaml
│   └── habitat_test_smoke.yaml
├── src/cmnav/
│   ├── cli.py                # 检查、运行、计划、采集和离线信息命令
│   ├── config.py             # 配置校验、覆盖与路径解析
│   ├── collection_config.py  # 采集配置与项目内数据路径
│   ├── collection.py         # 固定起点清单、采集循环与独立审计
│   ├── exploration_policy.py # 仅使用观测的深度探索规则
│   ├── trajectories.py       # 完整轨迹写入、读取与连续窗口
│   ├── memory.py             # 快照、任务副本与显式持久化
│   ├── diagnostics.py        # 依赖与数据静态检查
│   ├── envs/habitat_env.py    # Habitat 适配、观测和指标接口
│   ├── envs/exploration_env.py # 有限起点清单的无目标环境
│   ├── policies.py           # 最小随机策略
│   ├── runner.py             # episode 循环和运行汇总
│   └── recording.py          # 逐步记录、RGB-D 与预览
├── data/                     # 本地数据资源
├── scripts/
│   ├── prepare_mp3d_data.py   # MP3D 单场景与任务准备
│   └── download_test_data.py  # 公开测试数据下载
├── docs/setup.md
└── outputs/                  # 运行产物
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

新增模块已通过小规模验证，首批 10×500 开发采集及质量分析也已完成。下一步讨论降低探索碰撞与长时间停留的方法，再接入视觉状态表示、运动估计和 CML 地图更新。正式训练与评测前还需要准备更多场景、冻结正式采集预算及数据版本，并明确训练和评测任务划分。

第一批实验研究探索经验如何支持新目标导航，暂不设计专门的组合规则采集。代码提供持续记忆接口；首轮正式实验按 [已确认协议](docs/trajectory_protocol_v1.md) 采用任务内更新、任务间恢复快照。

# CMNav 数据准备

MP3D 已获得访问资格，用户已提供官方 `download_mp.py` 下载入口。CMNav 的场景、任务、下载缓存和数据来源清单统一保存在本目录。

一次 PointNav 运行需要两类文件：场景资源用于渲染和运动，episode 文件指定起点、目标和场景引用。拥有其中一类不足以运行任务。

## 自采轨迹与场景记忆

新增模块的默认数据目录如下；小规模采集验证数据保存在 `validation/collection_runs/`，测试临时数据也位于 `validation/`。目前尚未构建 CML 场景记忆：

- `collection_plans/`：固定起点清单，包含仅供环境初始化和审计使用的位姿。
- `trajectories/<run_id>/trajectories/`：完整 RGB-D、动作转移和可校验的轨迹元数据。
- `trajectories/<run_id>/audit/`：模拟器真值、起点清单、完整配置与运行来源。
- `memory/`：建议的不可变场景快照和任务生命周期记录目录。

具体命令与接口见 [轨迹采集](../docs/trajectory_collection.md) 和 [场景记忆](../docs/memory.md)。学习读取器不加载 `audit/`；数据划分先以完整轨迹或场景分组，再生成连续窗口。

首批单场景 10×500 开发数据位于 `trajectories/20261021T221704Z_003d1b95/`，共 5,010 次观测，完整性和读取审计通过。该批次的碰撞和重复访问诊断见 [开发采集记录](../docs/mp3d_pilot_10x500.md)，尚未作为正式训练数据版本冻结。

## MP3D 的最小准备

`configs/mp3d_smoke.yaml` 默认选择训练集中的 `17DRP5sb8fy`。本阶段使用 Habitat 官方 MP3D 示例包中的这个真实房屋，以及官方 PointNav v1 的对应训练任务分片。它只覆盖一个场景，后续训练与评测需要继续准备其他场景和独立任务划分。

```text
data/
├── scene_datasets/
│   └── mp3d/
│       └── 17DRP5sb8fy/
│           ├── 17DRP5sb8fy.glb
│           └── ...                       # 保留官方场景资源中的相关文件
├── datasets/
│   └── pointnav/mp3d/v1/train/
│       ├── train.json.gz
│       └── content/17DRP5sb8fy.json.gz
├── downloads/matterport/              # 官方脚本、场景清单、下载压缩包
└── manifests/mp3d_smoke.json           # URL、大小、SHA256、提取成员
```

从 CMNav 根目录准备本阶段数据：

```bash
.env/bin/python scripts/prepare_mp3d_data.py
```

脚本下载两个官方归档并仅提取上面的单场景资源与训练任务分片：

- [MP3D example v1.1](https://dl.fbaipublicfiles.com/habitat/mp3d/mp3d_example_v1.1.zip)：67,765,004 字节，包含 GLB、NavMesh、房屋/语义资源、场景配置和使用条款。
- [MP3D PointNav v1](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip)：418,680,926 字节，保留完整任务压缩包供后续使用，当前只安装 `17DRP5sb8fy` 的 train 分片。

来源入口见 [Habitat-Sim v0.3.3 数据说明](https://github.com/facebookresearch/habitat-sim/blob/v0.3.3/DATASETS.md)和 [Habitat-Lab v0.3.3 数据说明](https://github.com/facebookresearch/habitat-lab/blob/v0.3.3/DATASETS.md)。脚本检查归档大小和提取成员的 ZIP CRC，清单记录本次文件的 SHA256；该 SHA256 是本地来源记录，不冒充发布方提供的校验值。已有同内容文件会复用，不同内容文件不会覆盖。

用户提供的官方原始脚本保存在 `downloads/matterport/download_mp.py`。它使用 Python 2，`--task_data habitat` 指向 Habitat 转换包 `v1/tasks/mp3d_habitat.zip`；任务包下载后仍会提示继续下载原始扫描数据。当前 Python 3.9 环境不直接执行此旧脚本。本阶段采用上面的官方单场景包；扩展到完整 MP3D 时再按已授权流程获取 Habitat 包，不需要先下载 1.3 TB 原始图像、深度等全部数据。

配置的 `dataset.scenes_dir` 是 `data/scene_datasets`，不是单个场景目录。Habitat 根据 episode 中的 `scene_id` 查找资源；例如 `data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb` 会映射到 `scenes_dir` 下对应的 MP3D 文件。按本项目约定，后续新增数据也继续存放在 CMNav 的 `data/` 内。

MP3D 配置还指定 `dataset.scene_dataset_config`，读取包内的 `mp3d.scene_dataset_config.json`，明确场景坐标约定、着色和关联资源。适配器同时将该路径应用到加载后的 episodes，因为 Habitat-Lab 0.3.3 初始化时会优先使用 episode 中的场景配置。原始任务文件保持官方内容。

官方 PointNav 文件可能按场景分片。主文件中的 `episodes` 为空不一定代表数据缺失；还需保留其引用的 `content/` 文件。配置 `scene_ids` 应与任务文件中的场景名称一致。第一轮建议使用明确的单场景列表，便于完整检查。

准备后从 CMNav 根目录执行：

```bash
.env/bin/python -m cmnav doctor --config configs/mp3d_smoke.yaml --output outputs/doctor_mp3d.json
.env/bin/python -m cmnav run --config configs/mp3d_smoke.yaml
```

`doctor` 会读取选中的任务文件并检查场景路径；它不检查 GLB 是否能够渲染，也不初始化 NavMesh。实际加载结果由 `run` 验证。

## 公开 Habitat 测试数据

MP3D 下载完成之前，可用公开 Habitat 测试资源验证渲染和记录流程。`configs/habitat_test_smoke.yaml` 使用 `dataset.name: habitat_test`，选中 `skokloster-castle`：

```text
data/
├── scene_datasets/
│   └── habitat-test-scenes/
│       ├── skokloster-castle.glb
│       └── ...
└── datasets/
    └── pointnav/
        └── habitat-test-scenes/
            └── v1/train/
                └── train.json.gz
```

从 CMNav 根目录执行 `.env/bin/python scripts/download_test_data.py` 可下载这组公开测试资源。脚本的默认输出目录为 CMNav 的 `data/`。场景资源固定到公开资源库提交 `910c783fb954da8497ea5f811b843a76590ddddc`，任务使用官方测试 PointNav 数据。运行方式见主 README。公开测试场景的产物只用于环境检查，不能报告为 MP3D 结果。

原始场景、任务数据、下载脚本中的授权信息和运行输出均应留在本地，不作为代码文件提交。需要分享一次实验时，先检查产物与数据许可，再选择适合分享的记录。

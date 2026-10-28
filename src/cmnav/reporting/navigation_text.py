"""Navigation and teacher-subgoal prose templates from prepared diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from cmnav.analysis.report_contracts import MODES, INTERVALS
from cmnav.reporting.labels import GCML_LABELS as LABELS, MODE_LABELS


def failure_label(stage: dict[str, Any] | None) -> str:
    if stage is None:
        return "—"
    failure = stage["first_failure"]
    if failure is None:
        return "无"
    ordinal = failure.get("stage_ordinal")
    label = f"第 {ordinal} 段→节点 {failure['goal_node']}" if ordinal is not None else "阶段外连接/结束"
    return f"{label}：{failure.get('end_reason', failure.get('status', 'unknown'))}"


def gcml_markdown(run: Path, data: dict[str, Any], output: Path) -> str:
    summary, tasks = data["summary"], data["tasks"]
    failed_plans = [row for row in data["rows"] if row["end_reason"] == "no_plan"]
    before_hops = sum(row["graph_hops_completed"] == 0 for row in failed_plans)
    after_hops = len(failed_plans) - before_hops
    lines = ["# GCML 图导航方法对比", "", f"运行目录：`{run}`", "",
             "单场景、真值状态关联、冻结经验图、共享官方局部执行器；这是图覆盖条件下的开发诊断。",
             f"所有方法读取同一份包含 {len(tasks)} 个任务的清单，配置环境动作预算为每任务 {data['config']['max_steps']}。"
             "报告核对各方法的 episode、实际起点和目标一致；任务如何生成以原清单与协议为准。", "",
             "## 1. 全部任务结果", "",
             "| 任务 | 方法 | 成功 | SPL | 环境步数 | 图边完成/尝试 | 结束原因 |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for stats in data["rows"]:
        i, method = stats["task_index"], stats["method"]
        lines.append(f"| {i+1} / {stats['episode_id']} | {LABELS.get(method, method)} | {int(stats['success'])} | "
                     f"{stats['spl']:.3f} | {stats['steps']} | "
                     f"{stats['graph_hops_completed']}/{stats['graph_hops_attempted']} | {stats['end_reason']} |")
    lines += ["", "## 2. 方法汇总", "", "| 方法 | 成功数 | 平均 SPL | 平均环境步数 |",
              "| --- | ---: | ---: | ---: |"]
    for method, value in summary["methods"].items():
        lines.append(f"| {LABELS.get(method,method)} | {round(value['success']*value['episodes'])}/{value['episodes']} | "
                     f"{value['spl']:.4f} | {value['mean_steps']:.1f} |")
    lines += ["", f"保存结果中共有 {len(failed_plans)} 次 `no_plan`：{before_hops} 次尚未完成图边，"
              f"{after_hops} 次已完成至少一条图边后停止。`no_plan` 的少量动作是提前终止，不能解释为效率优势；"
              "是否已实际行走应结合图边完成数和逐步执行日志判断。", "",
              "## 3. 想象诊断", "", "| 方法 | 决策次数 | 候选数 | 预测到达 | 连续合法 | 图审计到达 | 虚假到达 | 想象转移数 | 规划秒数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, value in summary["methods"].items():
        info = value.get("gcml_planning")
        if info:
            lines.append(f"| {LABELS.get(method,method)} | {info['decisions']} | {info['candidates']} | "
                         f"{info['predicted_arrived_candidates']} | {info['continuous_valid_candidates']} | "
                         f"{info['graph_arrived_candidates']} | {info['false_predicted_arrival_candidates']} | "
                         f"{info['model_transitions']} | {info['observed_planner_call_seconds']:.2f} |")
    lines += ["", "GCML greedy 不生成想象候选，零候选不能解释成候选生成失败。图审计在规划完成后独立运行，未用于学得 G 的选路。",
              "", "## 4. 产物与边界", "",
              "- `training.json`：逐轮误差、矩阵范数、更新尺度与稳定性记录；`model.json`：Q/V/W/G。",
              "- `supported_tasks.json`：任务清单、图指纹与覆盖筛选记录；原始计划以运行内 `plan.json` 为准。",
              "- `evaluation/<method>/task_*/planning.jsonl`：全部实际决策及候选摘要。",
              "- `gcml_candidates.jsonl.gz`：逐候选逐步想象轨迹、读出、可用性、噪声和独立图审计。",
              "- `execution.jsonl`、`steps.jsonl`：全部实际动作、位姿、碰撞和环境指标。",
              "- `report_implementation.json`：本次分析源码快照、输入文件指纹及分析前后一致性核对。",
              "- GCML graph affordance 使用经验图可用边；其成功不能当作 G 已学好的证据。",
              "- 这不是 RGB-D 视觉导航结果、跨场景泛化实验或严格组合能力验证。", "",
              "## 5. 各任务实际路线图", ""]
    for i in range(len(tasks)):
        lines.append(f"![任务 {i+1} 实际执行路线]({output / f'task_{i:04d}_routes.png'})\n")
    return "\n".join(lines) + "\n"


def subgoal_markdown(run: Path, data: dict[str, Any], comparison: dict[str, Any],
                     output: Path, plots: list[Path]) -> str:
    config = data["config"]
    max_steps = data["max_steps"]
    aggregates = data["aggregates"]
    rows = data["rows"]
    inputs = data["inputs"]
    lines = ["# GCML 教师分阶段目标对比", "", f"运行：`{run}`。", "",
             f"本实验复用同一 GCML checkpoint、同一经验图与 {len(data['tasks'])} 个冻结任务。教师按既定图路径提供阶段目标；它额外提供了正确路线信息，结果不能解释成模型已经能自主提出子目标。",
             "学得 G 和经验图可用性辅助分别报告。终点直达是本轮同一流程下的对照，阶段间连续使用环境状态及总动作预算，不把每一段重置成独立任务。", "",
             "## 1. 固定条件与分母", "",
             f"- 每任务总预算：**{max_steps}** 个环境动作；所有阶段共同使用该预算。",
             f"- 每次决策：K={config['gcml_candidates']}，H={config['gcml_horizon']}，噪声 σ={config['gcml_noise_scale']}；每次最多 {config['gcml_candidates'] * config['gcml_horizon']} 次想象转移。",
             f"- 模型 SHA-256：`{comparison['model_sha256']}`。报告脚本只读取保存结果，没有训练、调参或追加想象。",
             f"- 已核对模型、经验图、任务清单与原运行的文件指纹；{len(rows)} 次任务参数均保持不变，{comparison['baseline_controls_reproduced']} 次终点直达对照均重现原运行。",
             "- 阶段数统一写为 **完成 / 已尝试 / 全部预定**。完成÷已尝试存在阶段选择偏差：早期失败会使后续阶段无法进入样本。完成÷全部预定同时报告。",
             "- 终点直达没有教师阶段，阶段数记为“—”。整任务成功与 SPL 始终采用环境对最终目标的原始评价。", "",
             "## 2. 整任务与阶段汇总", "",
             "| 目标间隔 | 可用性模式 | 成功任务 | 平均 SPL | 平均环境步 | 阶段完成/尝试/预定 | 完成÷尝试 | 完成÷预定 | no_plan 任务 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for interval in INTERVALS:
        for mode in MODES:
            value = next(item for item in aggregates if item["method"] == mode and item["interval"] == interval)
            stages = value["stages"]
            counts = "—" if stages is None else f"{stages['completed']}/{stages['attempted']}/{stages['total']}"
            conditional = "—" if stages is None or stages["completion_given_attempted"] is None else f"{100 * stages['completion_given_attempted']:.1f}%"
            unconditional = "—" if stages is None else f"{100 * stages['completion_of_all_planned']:.1f}%"
            label = "终点直达" if interval is None else f"每 {interval} 条图边"
            lines.append(f"| {label} | {MODE_LABELS[mode]} | {value['successes']}/{value['tasks']} | {value['mean_spl']:.4f} | "
                         f"{value['mean_steps']:.1f} | {counts} | {conditional} | {unconditional} | {value['no_plan_tasks']} |")
    lines += ["", "## 3. 计算量与候选", "",
              f"计算量是各条件全部 {len(data['tasks'])} 个任务的合计；增加阶段会改变重规划次数，不能只按相同 K/H 宣称整任务计算预算相等。",
              "耗时为保存的模型规划调用墙钟时间，包含实现和机器因素，不包含完整仿真及所有日志开销。少步 `no_plan` 是提前失败，不能当作导航效率优势。", "",
              "| 条件 | 决策 | 候选 | 预测到达候选 | 图审计到达候选 | 非法候选 | 想象转移 | 模型规划秒 | 外层调用秒 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for value in aggregates:
        plan = value["planning"]
        lines.append(f"| {value['condition']} | {plan['decisions']} | {plan['candidates']} | {plan['predicted_arrived_candidates']} | "
                     f"{plan['graph_arrived_candidates']} | {plan['invalid_candidates']} | {plan['model_transitions']} | "
                     f"{value['planner_seconds']:.3f} | {value['observed_planner_call_seconds']:.3f} |")
    lines += ["", "候选图审计发生在模型选择之后；教师阶段目标本身已使用图路线信息，这是本实验明确增加的权限。", "",
              f"## 4. 全部 {len(rows)} 次任务执行", "",
              "表中阶段编号从 1 开始；JSON 保留评测器的原始 index、节点、起止步数与状态。若提前经过原始最终目标，环境成功可与剩余教师阶段未完成并存；阶段记录和整任务成功分别计数。", "",
              "| 任务/episode | 条件 | 成功 | SPL | 环境步 | 图边完成/尝试 | 阶段完成/尝试/预定 | 首次失败阶段 | 结束原因 | 想象转移 | 规划秒 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for row in rows:
        stages = row["stages"]
        counts = "—" if stages is None else f"{stages['completed']}/{stages['attempted']}/{stages['total']}"
        lines.append(f"| {row['task_index'] + 1}/{row['episode_id']} | {row['condition']} | {int(row['success'])} | {row['spl']:.4f} | "
                     f"{row['steps']} | {row['graph_hops_completed']}/{row['graph_hops_attempted']} | {counts} | {failure_label(stages)} | "
                     f"{row['end_reason']} | {row['planning']['model_transitions']} | {row['observed_planner_call_seconds']:.3f} |")
    lines += ["", "## 5. 实际路线图", "",
              "每图两行分别是学得 G、经验图可用性辅助；四列依次是终点直达、每 4/8/16 条图边一个教师阶段目标。",
              "灰色为经验图；**蓝线只连接真实仿真执行记录中的位姿**；橙色空心圆及数字表示全部预定教师目标，包括尚未尝试的目标，不代表已经走过的路线。黑点是初始位置、绿星是最终环境目标、蓝叉是最后记录位置。",
              "图为世界 X/Z 投影；不同高度可能重叠，图中注明图节点的高度跨度。", ""]
    for index, plot in enumerate(plots):
        lines += [f"### 任务 {index + 1}", "", f"![任务 {index + 1} 真实执行路线与教师目标]({plot})", ""]
    lines += ["## 6. 可复查产物", "",
              f"- [完整比较 JSON]({output / 'comparison.json'})：全部数值、首失败阶段、原始阶段记录和输入文件指纹。",
              f"- [冻结教师目标清单]({inputs['teacher_subgoal_plan']})：本轮使用的目标计划。",
              "- `evaluation/<condition>/task_*/execution.jsonl` 与 `steps.jsonl`：每个实际动作、位姿和环境指标。",
              "- `planning.jsonl` 与 `gcml_candidates.jsonl.gz`：真实决策、想象候选与独立图审计。",
              "- 当前仍是同场景、真值状态、教师目标辅助开发诊断；不能据此宣称自主层级规划、RGB-D 视觉导航或跨场景泛化。", ""]
    return "\n".join(lines)

"""Human-readable structural reports for a frozen landmark hierarchy."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _number(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def _cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _distribution_row(label: str, summary: dict[str, Any]) -> str:
    return "| " + " | ".join([label] + [_number(summary[key]) for key in
                                          ("count", "min", "median", "mean", "p95", "max")]) + " |"


def _plot_structure(path: Path, hierarchy: dict[str, Any], audit: dict[str, Any],
                    graph_positions: list | None) -> None:
    """A static graph projection and distributions; plotting has no policy role."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    map_ax, length_ax, degree_ax, support_ax = axes.ravel()
    if graph_positions is not None:
        if len(graph_positions) != hierarchy["n_states"] or any(
            len(position) != 3 or any(not math.isfinite(float(value)) for value in position)
            for position in graph_positions
        ):
            raise ValueError("graph_positions must contain one finite XYZ position per low-level node.")
        landmarks = [graph_positions[node] for node in hierarchy["landmark_nodes"]]
        map_ax.scatter([position[0] for position in graph_positions],
                       [position[2] for position in graph_positions],
                       color="#ccd3dc", s=10, label="Low-level graph node", zorder=1)
        segments = [((landmarks[source][0], landmarks[source][2]),
                     (landmarks[target][0], landmarks[target][2]))
                    for source, target in hierarchy["edges"]]
        map_ax.add_collection(LineCollection(segments, colors="#4b759c", linewidths=.65, alpha=.4, zorder=2))
        # Arrowheads convey directed relations; the lines do not denote physical paths.
        for start, end in segments:
            map_ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                            mutation_scale=6, linewidth=.35,
                                            color="#4b759c", alpha=.5, zorder=2))
        map_ax.scatter([position[0] for position in landmarks],
                       [position[2] for position in landmarks],
                       s=22, color="#cc6927", label="Landmark", zorder=3)
        if len(landmarks) <= 60:
            for index, position in enumerate(landmarks):
                map_ax.annotate(str(index), (position[0], position[2]), xytext=(3, 3),
                                textcoords="offset points", fontsize=6, color="#6d320c")
        map_ax.set(xlabel="x (m)", ylabel="z (m)", title="Directed macro relations: x-z projection")
        map_ax.set_aspect("equal", adjustable="datalim")
        map_ax.legend(fontsize=8, loc="best")
    else:
        map_ax.axis("off")
        map_ax.text(.5, .5, "No graph positions supplied\nTopology statistics remain available", ha="center", va="center")
    lengths = [length for row in hierarchy["macro_edges"] for length in row["lengths"]]
    if lengths:
        low, high = min(lengths), max(lengths)
        length_ax.hist(lengths, bins=[value - .5 for value in range(low, high + 2)], color="#537faa", edgecolor="white")
    length_ax.set(title="Macro support segment lengths", xlabel="Low-level graph edges", ylabel="Support observations")
    degree = audit["topology"]["high_out_degree"]
    if degree:
        degree_ax.hist(degree, bins=[value - .5 for value in range(max(degree) + 2)], color="#6c9882", edgecolor="white")
    degree_ax.set(title="Landmark out-degree", xlabel="Distinct macro actions", ylabel="Landmarks")
    counts = [row["support_count"] for row in hierarchy["macro_edges"]]
    if counts:
        support_ax.hist(counts, bins=min(20, len(set(counts))), color="#b47b4b", edgecolor="white")
    support_ax.set(title="Macro-action support imbalance", xlabel="Observed support segments per macro action", ylabel="Macro actions")
    for axis in (length_ax, degree_ax, support_ax):
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Frozen hierarchy: structural audit only", fontsize=14)
    try:
        fig.savefig(path, dpi=180)
    finally:
        plt.close(fig)


def write_structure_report(output: Path, hierarchy: dict[str, Any], audit: dict[str, Any], *,
                           graph_positions: list | None = None) -> Path:
    """Write a fresh report and figure; existing report products are never replaced.

    The workflow saves hierarchy.json and structure_audit.json before calling
    this function, and is responsible for input identities and source archives.
    """
    output = Path(output)
    report_path = output / "report.md"
    figure_path = output / "hierarchy_structure.png"
    for path in (report_path, figure_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing hierarchy report product: {path}")
    if audit.get("schema") != "cmnav.landmark_hierarchy_audit.v1":
        raise ValueError("Unsupported hierarchy audit schema.")
    if hierarchy.get("schema") != "cmnav.landmark_hierarchy.v1":
        raise ValueError("Unsupported hierarchy data schema.")
    output.mkdir(parents=True, exist_ok=True)
    size, topology = audit["size"], audit["topology"]
    attachments, tasks = audit["attachments"], audit["task_summary"]
    support, replay = audit["macro_support"], audit["high_replay"]
    config = hierarchy["config"]
    lines = [
        "# Landmark 层次结构审计", "",
        "级别：开发结构诊断。使用已有真值辅助经验图上的连续重放序列；本报告没有训练模型、运行模型规划或执行 Habitat 导航。",
        "Landmark 是已有细层节点的子集，不代表语义房间或区域。下文的一步是一条有向细层图边，不是一次 MOVE_FORWARD 或转向。",
        "", "## 1. 构建与冻结约定", "",
        "Landmark 由固定顺序经验的形成前缀选择；集合冻结后再扫描全部训练经验提取宏转移与接入/接出关系。评测任务只用于冻结后的审计，没有按其起终点补点或补边。",
        "每个非零宏转移和接入/接出关系都有同一条重放序列内的连续支持片段；reset 边界不产生转移。这些支持不是新采集的完整物理执行轨迹。",
        "", "```json", json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "入口支持方向是普通节点 → landmark；出口支持方向是 landmark → 普通节点。Landmark 自身保留零步身份关系。",
        "", "## 2. 结构规模与支持情况", "",
        "| 指标 | 数值 |", "| --- | ---: |",
        f"| 细层节点 / 有向边 | {size['low_nodes']} / {size['low_edges']} |",
        f"| Landmark / 宏边 | {size['landmarks']} / {size['macro_edges']} |",
        f"| 保留节点比例 / 边数比例 | {_number(size['node_fraction_retained'])} / {_number(size['edge_fraction_retained'])} |",
        f"| 每 landmark 对应的细层节点数 | {_number(size['nodes_per_landmark'])} |",
        f"| 宏边支持片段总数 | {support['observations']} |",
        f"| 仅有一次支持的宏边数 | {len(support['single_support_actions'])} |",
        f"| 高层连续序列 / 转移数 | {replay['sequences']} / {replay['transitions']} |", "",
        "节点与边数比例只是结构规模，不等于计算节省或模型压缩收益。重复重放支持次数不等于独立环境采集次数。", "",
        "| 分布 | 数量 | 最小 | 中位数 | 均值 | P95 | 最大 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _distribution_row("每次宏支持的细层边长度", support["steps_per_observation"]),
        _distribution_row("每条宏边的支持次数", support["support_count_per_edge"]),
        _distribution_row("每条宏边的经验长度中位数", support["median_steps_per_edge"]),
        _distribution_row("每条高层连续序列的转移数", replay["transitions_per_sequence"]), "",
    ]
    counters = hierarchy.get("counters", {})
    if counters:
        lines += ["经验利用与排除记录如下；完整片段来源保存在 `hierarchy.json`。", "",
                  "```json", json.dumps(counters, ensure_ascii=False, indent=2, sort_keys=True), "```", ""]
    lines += [
        "## 3. 有向连通性与两端覆盖", "",
        "| 图 | 弱连通分量 | 强连通分量 | 无出边节点 | 孤立节点 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("low", "细层"), ("high", "高层")):
        row = topology[key]
        lines.append(f"| {label} | {row['weak_count']} | {row['strong_count']} | {len(row['sink_nodes'])} | {len(row['isolated_nodes'])} |")
    lines += [
        "", f"高层弱分量大小：`{topology['high']['weak_sizes']}`；强分量大小：`{topology['high']['strong_sizes']}`。",
        f"有向可达 landmark 对（含零步身份关系）：{topology['reachable_ordered_landmark_pairs_including_identity']} / {topology['all_ordered_landmark_pairs_including_identity']}。",
        "有向图中处于同一弱分量不代表能沿给定方向到达。无出边节点可能是合法终点，其 G 训练标签应为全零；不能因此直接删除。", "",
        "| 覆盖类型 | 细层节点数 / 总数 | 比例 |", "| --- | ---: | ---: |",
        f"| 可接入 landmark | {attachments['entry_nodes']} / {size['low_nodes']} | {_number(attachments['entry_fraction'])} |",
        f"| 可由 landmark 接出 | {attachments['exit_nodes']} / {size['low_nodes']} | {_number(attachments['exit_fraction'])} |",
        f"| 两者都有 | {attachments['both_nodes']} / {size['low_nodes']} | {_number(attachments['both_fraction'])} |",
        f"| 两者皆无 | {attachments['neither_nodes']} / {size['low_nodes']} | — |", "",
        "这些覆盖率穷举全部已保存的局部关系，不受后续入口/出口候选数和模型想象预算限制，因此是当前支持数据的结构覆盖上界。", "",
        "## 4. 开发任务结构审计", "",
        f"本次完整读取 {tasks['count']} 个既有开发任务。底层图可达 {tasks['bottom_reachable']} 个；存在“接入 → 宏图 → 接出”支持组合 {tasks['structurally_supported']} 个。",
        "所有任务均保留。它们已反复用于开发，不是独立泛化评测；本表不计模型导航成功率。", "",
        "| 任务 / episode | 起点 → 终点 | 底层最少边 | 入口 / 出口数 | 可连通端点对 / 全部对 | 最低估计总边数 | 结构状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    labels = {"no_entry": "无接入支持", "no_exit": "无接出支持", "high_disconnected": "高层不连通",
              "structurally_supported": "有完整结构支持"}
    for task in audit["tasks"]:
        best = task["best_median_cost_pair_audit_only"]
        estimated = best["estimated_total_steps"] if best else None
        lines.append(
            f"| {_cell(task['task_index'])} / {_cell(task['episode_id'])} | {task['start_node']} → {task['goal_node']} | "
            f"{_number(task['bottom_minimum_hops'])} | {task['entry_candidate_count']} / {task['exit_candidate_count']} | "
            f"{task['reachable_endpoint_pairs']} / {task['total_endpoint_pairs']} | {_number(estimated)} | {labels[task['status']]} |")
    lines += [
        "", "估计总成本 = 接入片段长度中位数 + 宏路径各边长度中位数之和 + 接出片段长度中位数。它用于结构诊断；不等于模型实际行走成本，也不保证任一具体原始轨迹具有这个完整组合。",
        "起终点对应同一 landmark 时允许零条宏边。若普通节点之间存在短底层路径，但当前层次结构不支持它，只记录直接路径诊断，不据此偷偷增加模型候选或教师子目标。", "",
        "### 各任务的最低估计成本组合（仅供审计）", "",
    ]
    for task in audit["tasks"]:
        best = task["best_median_cost_pair_audit_only"]
        if best:
            actions = best["minimum_median_cost_macro_actions"]
            route = [best["entry_landmark"]] + [hierarchy["edges"][action][1] for action in actions]
            low_route = [hierarchy["landmark_nodes"][node] for node in route]
            lines.append(f"- 任务 {task['task_index']}：入口/出口细层节点 {best['entry_node']} / {best['exit_node']}；"
                         f"landmark 高层 ID 路线 `{route}`，对应细层节点 `{low_route}`；宏动作 `{actions}`；"
                         f"接入/宏段/接出估计成本 {_number(best['entry_median_steps'])} / {_number(best['macro_median_cost'])} / {_number(best['exit_median_steps'])}。")
        else:
            missing = ", ".join(task["missing_attachments"]) or "none; macro graph disconnected"
            lines.append(f"- 任务 {task['task_index']}：{labels[task['status']]}；缺失关系 `{missing}`。"
                         f"底层可达={task['bottom_reachable']}；存在不超过接入预算的直接底层路径="
                         f"{task['direct_short_path_within_attachment_horizon_audit_only']}（仅诊断）。")
    lines += [
        "", "## 5. 结构图与解释边界", "",
        "![Landmark 结构与支持分布](hierarchy_structure.png)", "",
        "左上图为 x-z 平面投影，丢弃 y 高度；跨楼层节点可能重叠。箭头只连接宏转移端点，不代表穿墙直线或实际执行路线。标注数字为高层 landmark ID；完整低/高 ID 映射在 `hierarchy.json`。",
        "", "当前结构审计只回答：冻结经验是否提供分阶段路线的支持。它没有回答 G 是否拟合充分、高层能否规划、细层能否完成选定阶段，或总体成功率是否提升。",
        "接下来应固定同一细层 checkpoint 和本次数据版本，先检查高层单宏边预测与全节点 G，再比较平坦 GCML、高层 Dijkstra + 细层 GCML、高层 GCML + 细层 GCML、底层 Dijkstra。模型候选仍按统一总预算计数，执行失败保留其具体阶段。",
        "高层/底层最短路和真实边合法性只用于审计或明确标记的图搜索对照；不能实时替学得模型排除未来非法动作。", "",
        "## 产物", "",
        "- `hierarchy.json`：冻结 landmark、宏动作、两端关系、支持片段和抽取参数。",
        "- `structure_audit.json`：连通性、覆盖率、完整逐任务审计与结构最短路。",
        "- `hierarchy_structure.png`：本报告静态图。",
        "- 运行配置、输入指纹、源码快照和状态由所属 workflow 保存；本报告不代替运行来源记录。", "",
    ]
    _plot_structure(figure_path, hierarchy, audit, graph_positions)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


_METHOD_LABELS = {
    "flat_gcml": "平坦 GCML",
    "hierarchical_dijkstra": "高层 Dijkstra + 细层 GCML",
    "hierarchical_gcml": "高层 GCML + 细层 GCML",
    "dijkstra": "底层 Dijkstra",
}


def _plot_execution_routes(path: Path, rows: list[dict[str, Any]], *, seed: int,
                           graph_positions: list, graph_edges: list, hierarchy: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if len(graph_positions) != hierarchy["n_states"] or any(
        len(position) != 3 or any(not math.isfinite(float(value)) for value in position)
        for position in graph_positions
    ):
        raise ValueError("Graph route plots require one finite XYZ position per low-level node.")
    positions = [(position[0], position[2]) for position in graph_positions]
    background = [(positions[source], positions[target]) for source, target in graph_edges]
    landmarks = [positions[node] for node in hierarchy["landmark_nodes"]]
    task_order = list(dict.fromkeys(row["task_index"] for row in rows))
    lookup = {(row["method"], row["task_index"]): row for row in rows if row["seed"] == seed}
    methods = list(_METHOD_LABELS)
    titles = ["Flat GCML", "High Dijkstra + low GCML", "High GCML + low GCML", "Low graph Dijkstra"]
    phase_colors = {"full_goal": "#1f77b4", "direct": "#1f77b4", "entry": "#9e48a0",
                    "macro": "#16836b", "exit": "#b96b26"}
    fig, axes = plt.subplots(max(1, len(task_order)), len(methods), figsize=(18, max(3.8, 3.8 * len(task_order))),
                             squeeze=False, constrained_layout=True)
    for row_index, task_index in enumerate(task_order):
        for column, method in enumerate(methods):
            axis = axes[row_index, column]
            axis.add_collection(LineCollection(background, colors="#d6dce3", linewidths=.55, zorder=1))
            axis.scatter([position[0] for position in landmarks], [position[1] for position in landmarks],
                         color="#d4a879", s=9, marker="D", zorder=2)
            row = lookup.get((method, task_index))
            if row is None:
                axis.text(.5, .5, "No saved row for this seed", ha="center", transform=axis.transAxes)
                axis.set_title(f"{titles[column]} | task {task_index}", fontsize=10)
            else:
                trace = row["trace"]
                segments = [(positions[step["source"]], positions[step["target"]]) for step in trace]
                if segments:
                    axis.add_collection(LineCollection(segments,
                                                       colors=[phase_colors.get(step["phase"], "#495468") for step in trace],
                                                       linewidths=2.3, zorder=3))
                start, goal, final = positions[row["start"]], positions[row["goal"]], positions[row["final_node"]]
                axis.scatter([start[0]], [start[1]], s=60, marker="o", facecolors="white", edgecolors="#111111", zorder=5)
                axis.scatter([goal[0]], [goal[1]], s=90, marker="*", color="#d62728", zorder=6)
                axis.scatter([final[0]], [final[1]], s=32, marker="x", color="#111111", zorder=7)
                status = "success" if row["success"] else "failed"
                axis.set_title(f"{titles[column]} | task {task_index}\n{status}; {row['graph_steps']} edges; {row['reason']}", fontsize=9)
            axis.autoscale_view()
            axis.set_aspect("equal", adjustable="datalim")
            axis.set(xlabel="x (m)", ylabel="z (m)")
            axis.tick_params(labelsize=8)
    fig.suptitle(f"Actual abstract-graph routes, planning seed {seed}\n"
                 "Circle: start; star: goal; cross: final; diamonds: landmarks. x-z projection, y omitted.", fontsize=12)
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def write_experiment_report(output: Path, summary: dict[str, Any], *, graph_positions: list,
                            graph_edges: list, hierarchy: dict[str, Any]) -> Path:
    """Render all saved methods/seeds/tasks and first-seed actual graph routes.

    Each row is a run_graph_task result tagged by the workflow with task_index
    and episode_id. This function never reruns or modifies any policy outcome.
    """
    output = Path(output)
    rows = summary["rows"]
    if not rows:
        raise ValueError("An experiment report requires saved task results.")
    identities = [(row["method"], row["seed"], row["task_index"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate method/seed/task rows cannot enter a comparison report.")
    if any(row["method"] not in _METHOD_LABELS for row in rows):
        raise ValueError("Unknown method in hierarchy comparison report.")
    for row in rows:
        if not row["nodes"] or row["nodes"][0] != row["start"] or row["nodes"][-1] != row["final_node"]:
            raise ValueError("Saved task node sequence does not match its endpoints.")
        if len(row["nodes"]) != row["graph_steps"] + 1 or len(row["trace"]) != row["graph_steps"]:
            raise ValueError("Saved task route length does not match its graph-step count.")
        if row["success"] != (row["final_node"] == row["goal"]):
            raise ValueError("Abstract graph success must mean actual goal-node arrival.")
    seeds = list(dict.fromkeys(row["seed"] for row in rows))
    task_ids = list(dict.fromkeys(row["task_index"] for row in rows))
    first_seed = seeds[0]
    report_path = output / "experiment_report.md"
    figure_path = output / f"execution_routes_seed_{first_seed}.png"
    for path in (report_path, figure_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite a hierarchy experiment report product: {path}")
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Landmark 层次 GCML：抽象图开发实验", "",
        f"结果范围：{len(task_ids)} 个既有开发任务，规划 seeds `{seeds}`，共 {len(rows)} 条方法/seed/任务运行记录。",
        (f"本报告从已有完整记录生成，没有重新执行策略。来源运行 `{summary['source_evaluation']}` 的原始状态为 `{summary['source_evaluation_status']}`；原始错误为 `{summary.get('source_error')}`。报告生成成功不改写来源运行状态。"
         if summary.get("report_only") else ""),
        "本轮只在已有抽象图上执行，成功判据为真实图节点 ID 到达目标；没有运行 Habitat 物理控制，也没有计算 Habitat SPL。相同任务的多个规划 seed 不是多个独立任务。",
        "Landmark 是已有节点，宏转移由连续图重放支持；不具有语义区域含义，也不构成新的物理轨迹采集。细层 checkpoint 在各方法中冻结并共享，高层训练来源和设置见本运行训练记录。", "",
        "## 1. 实际协议", "", "```json",
        json.dumps(summary["protocol"], ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "层次方法共用冻结的 landmark、宏边和两端关系。入口/出口候选从经验关系中按显式规则选取；保存的支持路径不直接作为细层执行动作。不同子查询的想象转移共同计入任务总预算。",
        "平坦 GCML 使用独立声明的完整目标候选数，层次查询使用每次查询的候选数；比较采用同一任务想象转移上限，实际消耗仍可能不同。不能把候选数或最大预算相等写成实际计算量相等。",
        "已知查询起点的出边限制属于当前特权图诊断权限，包括拟查询的 landmark 端点；未来想象状态由学得 G 决定。事后图审计用于统计非法候选和虚假到达，不据此替模型重新选择候选。",
        "高层 Dijkstra 与底层 Dijkstra 使用图搜索；其搜索展开数单独记录，不等价于神经模型转移数，也不声称两者 FLOPs 相同。",
        "阶段结束以实际节点到达判定；任务内模型参数不更新。经过其他 landmark 不自动改变目标；首轮失败不自动换入口、重新采样路线或恢复。", "",
        "## 2. 各方法、各规划 seed 的结果", "",
        "| 方法 | seed | 成功 / 实际任务记录 | 成功任务平均图边数 | 全部任务平均想象转移 | 图搜索展开总数 | 完成阶段数 / 完成宏阶段数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, label in _METHOD_LABELS.items():
        for seed in seeds:
            group = [row for row in rows if row["method"] == method and row["seed"] == seed]
            if not group:
                continue
            successful = [row for row in group if row["success"]]
            mean_steps = sum(row["graph_steps"] for row in successful) / len(successful) if successful else None
            mean_transitions = sum(row["model_transitions"] for row in group) / len(group) if group else None
            lines.append(f"| {label} | {seed} | {len(successful)} / {len(group)} | {_number(mean_steps)} | {_number(mean_transitions)} | "
                         f"{sum(row['graph_expansions'] for row in group)} | {sum(row['completed_stages'] for row in group)} / {sum(row['completed_macros'] for row in group)} |")
    lines += [
        "", "表中完成阶段 / 宏阶段是两个计数，不是成功分数的分子/分母。总图边数包含实际执行；失败前未走任何边时为 0，不能据此理解为路径更短。底层 Dijkstra 若在不同 seed 上重复，属于同一确定性对照的重复记录。", "",
        "## 3. 全部逐任务结果", "",
        "| 方法 | seed | 任务 / episode | 起点 → 终点 | 成功 | 图边数 / 重访数 | 终点节点 | 想象转移 / 图展开 | 完成阶段 / 宏阶段 | 失败阶段 | 终止原因 |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {_METHOD_LABELS[row['method']]} | {row['seed']} | {_cell(row['task_index'])} / {_cell(row['episode_id'])} | "
                     f"{row['start']} → {row['goal']} | {'是' if row['success'] else '否'} | {row['graph_steps']} / {row['node_revisits']} | "
                     f"{row['final_node']} | {row['model_transitions']} / {row['graph_expansions']} | {row['completed_stages']} / {row['completed_macros']} | "
                     f"{_cell(row['failure_phase'] or '—')} | {_cell(row['reason'])} |")
    lines += ["", "## 4. 失败与想象诊断", "",
              "| 方法 | 失败阶段及次数 | 预算阻止的查询数 | 候选数 | 非法候选 | 虚假到达候选 | 选中非法 / 选中虚假到达 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, label in _METHOD_LABELS.items():
        group = [row for row in rows if row["method"] == method]
        failure_counts = dict(sorted(Counter(row["failure_phase"] or "unclassified" for row in group if not row["success"]).items()))
        planning = {key: sum(row["planning_summary"][key] for row in group)
                    for key in ("candidates", "invalid_candidates", "false_predicted_arrivals",
                                "selected_invalid_candidates", "selected_false_predicted_arrivals")}
        lines.append(f"| {label} | `{json.dumps(failure_counts, ensure_ascii=False, sort_keys=True)}` | "
                     f"{sum(row['budget_blocked_queries'] for row in group)} | {planning['candidates']} | {planning['invalid_candidates']} | "
                     f"{planning['false_predicted_arrivals']} | {planning['selected_invalid_candidates']} / {planning['selected_false_predicted_arrivals']} |")
    lines += [
        "", "非法候选表示逐边图审计发现动作不连续/不合法；虚假到达表示模型预测到达但图审计没有到达，两者可能重叠。Dijkstra 没有神经想象候选，候选相关的 0 表示不适用。",
        "完整结果中的 `queries` 保留每次查询、候选审计与预算消耗，`stages` 保留阶段状态，`trace` 保留每步实际选边及来源查询。可以据此区分接入失败、高层无计划、宏边执行失败和接出失败。", "",
        "## 5. 执行路线图", "",
        f"![首个规划 seed 的全部任务路线]({figure_path.name})", "",
        f"图只展示本次记录顺序中的第一个规划 seed `{first_seed}`，不是挑选成功最多的 seed；其余 seed 的完整结果在上表和逐任务记录中。",
        "每行对应一个任务，每列一个方法。白圆为起点，红星为目标，黑叉为最终节点，菱形为 landmark。蓝色为平坦/直接阶段，紫色为接入，绿色为宏阶段，棕色为接出。",
        "路线是实际执行过的抽象图边；同一边重复执行可能在图上重叠，以图边数和重访数为准。x-z 投影丢弃 y 高度，不能代替物理楼层图或碰撞可行性检查。", "",
        "## 6. 解释范围与来源", "",
        "结构覆盖、高层学习、两端候选、细层规划与执行共同影响结果。高层 Dijkstra + 细层 GCML 是定位结构与细层执行问题的对照；只有在对应任务、seed、候选及预算设置下比较，才能判断高层 GCML 引入了哪些差异。",
        "本轮使用既有开发任务、一个场景及真值节点；不能据此宣称视觉导航、跨场景泛化、语义分层或独立组合泛化。即使训练损失很低或 G 在已知节点零误差，也不能单独证明长程导航成功。", "",
        "训练来源与元数据：", "", "```json",
        json.dumps(summary.get("training", {}), ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "冻结结构见 `hierarchy.json` 和结构报告 `report.md`；完整运行来源、结果路径与源码快照见本运行 manifest/summary。", "",
    ]
    _plot_execution_routes(figure_path, rows, seed=first_seed, graph_positions=graph_positions,
                           graph_edges=graph_edges, hierarchy=hierarchy)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

"""Reports for a frozen-model, paired hierarchy noise experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_LABELS = {
    "dijkstra": "底层 Dijkstra", "flat_gcml": "平坦 GCML",
    "hierarchical_dijkstra": "高层 Dijkstra＋细层 GCML",
    "hierarchical_gcml": "高层 GCML＋细层 GCML",
}


def _cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _number(value: Any) -> str:
    return "—" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)


def _counts(value: dict) -> str:
    return "、".join(f"`{_cell(key)}`: {count}" for key, count in sorted(value.items())) or "—"


def _plot_success(path: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"dijkstra": "Low graph Dijkstra (deterministic control)",
              "flat_gcml": "Flat GCML", "hierarchical_dijkstra": "High Dijkstra + low GCML",
              "hierarchical_gcml": "High GCML + low GCML"}
    conditions = summary["conditions"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = ("#3977a8", "#d48635", "#438c72", "#a467a5")
    for axis, method in zip(axes.ravel(), summary["methods"]):
        seeds = sorted({row["seed"] for condition in conditions for row in condition["method_seed"]
                        if row["method"] == method})
        width = .75 / len(seeds)
        max_tasks = 0
        for index, seed in enumerate(seeds):
            rows = [next(row for row in condition["method_seed"]
                         if row["method"] == method and row["seed"] == seed) for condition in conditions]
            max_tasks = max(max_tasks, *(row["count"] for row in rows))
            x = [i - .375 + width * (index + .5) for i in range(len(conditions))]
            bars = axis.bar(x, [row["successes"] for row in rows], width=width,
                            color=colors[index % len(colors)], label=f"seed {seed}")
            for bar, row in zip(bars, rows):
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .06,
                          f"{row['successes']}/{row['count']}", ha="center", va="bottom", fontsize=8)
        axis.set_xticks(range(len(conditions)))
        axis.set_xticklabels([f"C{i}\nL={condition['local_noise']:g}, H/F={condition['long_noise']:g}"
                              for i, condition in enumerate(conditions)], fontsize=8)
        axis.set(title=titles[method], ylabel="Successful development tasks", ylim=(0, max_tasks + .65))
        axis.set_yticks(range(max_tasks + 1))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8, loc="upper left")
    fig.suptitle("Frozen-model noise comparison: query-role proxies, not measured goal distances\n"
                 "L: entry/exit/macro; H/F: high GCML, full goal, direct goal. Reused controls are not new runs.",
                 fontsize=11)
    try:
        fig.savefig(path, dpi=180)
    finally:
        plt.close(fig)


def write_noise_comparison_report(output: Path, summary: dict) -> Path:
    """Write report.md and noise_success.png from a pure saved-result summary.

    Both products must be absent. Caller saves the machine-readable summary and
    runtime/config/source provenance; this function does not rerun evaluations.
    """
    if summary.get("schema") != "cmnav.hierarchy_noise_comparison.v1":
        raise ValueError("Unsupported hierarchy noise-comparison schema.")
    output = Path(output)
    report_path, figure_path = output / "report.md", output / "noise_success.png"
    for path in (report_path, figure_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite a noise-comparison product: {path}")
    conditions = summary["conditions"]
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 层次 GCML：冻结模型的噪声 2×2 对照", "",
        "级别：开发诊断。仅改变指定查询类型的高斯噪声标准差；模型、经验图、landmark、任务及其余推理规则由所属 workflow 冻结并核验。报告本身只汇总保存结果，不重新执行模型。",
        f"范围：{len(summary['task_indices'])} 个既有开发任务，规划 seeds `{summary['planning_seeds']}`。每条件 {summary['condition_row_count']} 条记录，表格共呈现 {summary['displayed_row_count']} 条；"
        f"按来源运行／方法／seed／任务去重后为 {summary['unique_source_method_seed_task_count']} 条来源记录。复用记录不是额外实验或独立样本。", "",
        "## 1. 条件定义与解释范围", "",
        "| 图中编号 | 条件 | 层次细层噪声 σL | 高层／平坦噪声 σH/F | 是否基线 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for index, condition in enumerate(conditions):
        lines.append(f"| C{index} | {_cell(condition['label'])} (`{_cell(condition['name'])}`) | {condition['local_noise']:g} | "
                     f"{condition['long_noise']:g} | {'是' if condition['name'] == summary['baseline'] else '否'} |")
    lines += [
        "", "σL 用于层次细层的接入候选、接出候选及宏边执行查询（entry_candidate / exit_candidate / macro_execution）；σH/F 用于全部高层 GCML 查询，以及细层的完整目标／直接目标候选查询（full_goal / direct_candidate）。direct_candidate 即使只有 8 步想象深度，也按最终目标调用归入 σH/F。底层 Dijkstra 无噪声参数；高层 Dijkstra 的图搜索不使用噪声，但该方法的 direct_candidate 仍受 σH/F 影响。",
        "这些类别是查询用途的代理，不是真实图距离分段。高层一条宏边与细层一条边也不是同一距离单位。本轮没有根据当前节点到目标的真值最短路动态设置噪声，因此不能证明“目标越近应加噪、越远应减噪”的单调规律。",
        "高斯加性噪声会改变分支采样；有效作用还取决于合法动作评分差距、出边数与终止规则。相同总想象转移上限不保证实际消耗相同，更不保证不同层模型的 FLOPs 相等。",
        "当前成功指实际抽象节点到达目标，不包含 Habitat 物理执行或视觉泛化。五个开发任务已反复分析，多个规划 seed 不是新的独立任务。", "",
        "## 2. 各条件、方法、seed 的成功与消耗", "",
        "| 条件 | 方法 | seed | 成功／任务 | 成功任务平均执行图边 | 平均／最大想象转移 | 完成阶段／宏阶段 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in conditions:
        for row in condition["method_seed"]:
            lines.append(f"| {_cell(condition['label'])} | {_LABELS[row['method']]} | {row['seed']} | {row['successes']}/{row['count']} | "
                         f"{_number(row['successful_graph_steps_mean'])} | {_number(row['model_transitions_mean'])}/{row['model_transitions_max']} | "
                         f"{row['completed_stages']}/{row['completed_macros']} |")
    lines += ["", "阶段／宏阶段列是两个完成次数，不是分子分母；其中宏阶段可能沿重复宏边出现。失败前执行较少不能解释为路径更优。", "",
              "![各条件每个规划 seed 的成功任务数](noise_success.png)", "",
              "## 3. 相同任务、seed 的配对变化", "",
              "| 条件 | 方法 | 配对记录 | 失败→成功 | 成功→失败 | 保持成功 | 保持失败 | 成功净变化 | 想象转移总量变化 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition in conditions:
        if condition["name"] == summary["baseline"]:
            continue
        for row in condition["methods"]:
            paired = row["paired_to_baseline"]
            lines.append(f"| {_cell(condition['label'])} | {_LABELS[row['method']]} | {paired['pairs']} | {paired['gained_success']} | "
                         f"{paired['lost_success']} | {paired['retained_success']} | {paired['retained_failure']} | "
                         f"{paired['net_success_change']:+d} | {paired['model_transitions_change']:+d} |")
    lines += ["", "### 成功状态改变的任务", "",
              "| 条件 | 方法 | seed／任务 | 改变 | 基线失败阶段／原因 | 本条件失败阶段／原因 | 基线→本条件想象转移 |",
              "| --- | --- | --- | --- | --- | --- | ---: |"]
    changes = 0
    for condition in conditions:
        if condition["name"] == summary["baseline"]:
            continue
        for group in condition["methods"]:
            for row in group["paired_to_baseline"]["changed_tasks"]:
                changes += 1
                old, new = row["before"], row["after"]
                label = "失败→成功" if row["change"] == "gained_success" else "成功→失败"
                lines.append(f"| {_cell(condition['label'])} | {_LABELS[row['method']]} | {row['seed']}/{row['task_index']} | {label} | "
                             f"{_cell(old['failure_phase'] or '—')} / {_cell(old['reason'])} | "
                             f"{_cell(new['failure_phase'] or '—')} / {_cell(new['reason'])} | "
                             f"{old['model_transitions']}→{new['model_transitions']} |")
    if not changes:
        lines.append("| — | — | — | 无成功状态改变 | — | — | — |")
    lines += ["", "完整配对记录、未改变任务及逐任务 trace 来源保存在机器可读汇总中。相同来源复用产生的零差异是预期控制，不能作为独立重复验证。", "",
              "## 4. 失败阶段、完成宏段与预算", "",
              "| 条件 | 方法 | 失败阶段计数 | 失败原因计数 | 完成宏阶段 | 预算阻止查询数 | 细／高层想象转移 | 图搜索展开 |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for condition in conditions:
        for row in condition["methods"]:
            planning = row["planning"]
            lines.append(f"| {_cell(condition['label'])} | {_LABELS[row['method']]} | {_counts(row['failure_phases'])} | "
                         f"{_counts(row['failure_reasons'])} | {row['completed_macros']} | {row['budget_blocked_queries']} | "
                         f"{planning['low_model_transitions']}/{planning['high_model_transitions']} | {row['graph_expansions_total']} |")
    lines += ["", "高层 Dijkstra 仅提供宏路线，实际宏阶段仍需细层 GCML 完成。汇总失败阶段不能替代逐查询诊断；同一个 no_plan 可能来自深度耗尽、回环或空候选等不同终止原因。", "",
              "## 5. 候选合法性、虚假到达与终止分布", "",
              "| 条件 | 方法 | 审计候选数 | 不同序列数之和 | 非法候选 | 虚假到达 | 选中非法／虚假到达 | 候选终止分布 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for condition in conditions:
        for row in condition["methods"]:
            planning = row["planning"]
            diversity = _number(planning['distinct_sequences_sum'])
            if planning['distinct_sequences_available_rows'] != row['count']:
                diversity += f"（覆盖 {planning['distinct_sequences_available_rows']}/{row['count']} 条）"
            lines.append(f"| {_cell(condition['label'])} | {_LABELS[row['method']]} | {planning['candidates']} | {diversity} | "
                         f"{planning['invalid_candidates']} | {planning['false_predicted_arrivals']} | "
                         f"{planning['selected_invalid_candidates']}/{planning['selected_false_predicted_arrivals']} | "
                         f"{_counts(planning['candidate_terminations'])} |")
    lines += [
        "", "非法候选是事后图审计发现动作不连续／不合法；虚假到达是模型预测到达但图审计没有到达，两者可能重叠。图审计不参与排除候选或修复计划。选中项统计查询层面的选择，不代表该查询一定进入最终执行路线。",
        "候选终止分布来自保存的 planning_summary，按当前模型日志口径计数。不同序列数之和先在每个查询内对完整动作序列去重，再把查询计数相加；它不是全局独立路线数，不能解释为独立任务样本数。旧行缺失该字段时显示缺失及覆盖数量，不补零；本报告未重新推导每步分支多样性。候选数增加或空集合减少本身都不能代替成功、非法想象与回环的联合判断。Dijkstra 候选相关的零表示不适用。", "",
        "## 6. 来源与复用", "",
        "| 条件 | 方法 | 来源运行 | 与基线复用相同来源 |", "| --- | --- | --- | --- |",
    ]
    for condition in conditions:
        for row in condition["methods"]:
            source = row["source"]
            link = f"[来源](<{source}>)" if source.startswith("/") else f"`{_cell(source)}`"
            same = "基线本身" if condition["name"] == summary["baseline"] else "是" if row["same_source_as_baseline"] else "否"
            lines.append(f"| {_cell(condition['label'])} | {_LABELS[row['method']]} | {link} | {same} |")
    lines += [
        "", "来源列表按方法保存，可识别仅受某一个噪声参数影响的方法在其他条件中的结果复用。全部跨条件复用关系保存在 summary 的 source_records；新来源路径表示来源不同，不单独证明输入模型相同，实际输入 SHA、冻结参数检查和运行状态由 workflow 保存。", "",
        "## 7. 本轮能回答什么", "",
        "这组 2×2 配对实验检验：固定当前模型、结构和五个开发任务时，增加层次细层噪声、降低高层／平坦噪声各自以及组合后的执行变化。若只看到某个条件更好，应同时看退步任务、失败阶段和实际消耗，不把最好单个 seed 当作稳定结论。",
        "由于本轮未按真实目标距离分层采样，也未按评分差距自适应设置噪声，结果不能证明距离与最佳噪声的单调关系。下一步若要验证该规律，应固定模型，在另外冻结的节点对上联合记录目标距离、合法出边评分差距及噪声响应。",
        "机器可读汇总额外保存每个方法／seed 的成功数差分交互量 S(高局部,低长程)−S(高局部,原长程)−S(原局部,低长程)+S(原局部,原长程)。这是这些配对记录上的描述量，不提供显著性或跨任务泛化证据。", "",
        "```json", json.dumps(summary["scope"], ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "产物：`report.md`、`noise_success.png`；机器可读结果、运行配置快照、日志、输入指纹及源码记录由所属 workflow 保存在同一 outputs 运行目录。", "",
    ]
    _plot_success(figure_path, summary)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

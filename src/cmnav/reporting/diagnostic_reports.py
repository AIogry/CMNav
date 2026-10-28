"""Text reports for offline mechanism and frozen-training diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo

def write_mechanism_report(analysis: dict[str, Any], output: Path) -> None:
    gating = analysis["gating"]
    training = {"epochs": analysis["epochs"]}
    run = Path(analysis["run"])
    threshold = gating["configured_threshold"]
    known_sources = gating["groups_at_configured_threshold"]["observed_sources"]["state_ids"]
    sinks = gating["groups_at_configured_threshold"]["observed_sinks"]["state_ids"]
    task_reports = analysis["tasks"]
    ratios = [row["top_two_gap_over_pairwise_noise_sd"] for row in analysis["branch_decisions"]
              if row["top_two_gap_over_pairwise_noise_sd"] is not None]
    from collections import Counter
    candidate_totals = Counter(analysis["candidate_totals"])
    selected_invalid = analysis["selected_false_arrival_candidates"]
    groups = gating["groups_at_configured_threshold"]
    def number(value: Any) -> str:
        return "—" if value is None else f"{value:.4g}"
    lines = [
        "# GCML 首轮机制诊断", "", f"运行：`{run.name}`。训练 {training['epochs']} 轮。",
        "", "本分析只读取已完成运行；没有训练、调参、重跑或把最短路标签反馈给模型。",
        "", "## 1. G 的监督范围与可行性识别", "",
        f"阈值为运行既定的 `{threshold}`。已观察出边状态共 {len(known_sources)} 个；汇点共 {len(sinks)} 个。",
        "汇点没有以训练转移源状态出现；其全负可行性标签没有被直接监督，不能与受监督源状态混为同一种误差。",
        "", "| 状态组 | 状态数 | TP | FP | FN | Precision | Recall | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("all_known_states", "全部"), ("observed_sources", "有出边源状态"), ("observed_sinks", "汇点")):
        row = groups[key]
        lines.append(f"| {label} | {row['states']} | {row['true_positive']} | {row['false_positive']} | {row['false_negative']} | {number(row['precision'])} | {number(row['recall'])} | {number(row['false_positive_rate'])} |")
    lines += ["", "Recall 在无正标签的汇点上未定义，记为 —。真值范围仅是已观察经验图。",
              "", "## 2. 首次规划是否采到最短路的第一条边", "",
              "允许等成本的多条正确首边；正确首边出现不代表整条候选最终到达。",
              "", "| 方法/任务 | 最短路首边集合 | 正确首边候选/总候选 | 最大合法前缀 | 最终结果 |",
              "|---|---|---:|---:|---|"]
    for row in task_reports:
        first = row["first_decision"]
        lines.append(f"| {row['task']} | {first['shortest_path_first_edges']} | {first['candidates_starting_with_correct_branch']}/{first['candidate_count']} | {first['maximum_valid_prefix_actions']} | {row['success']}; {row['end_reason']} |")
    lines += ["", "Dijkstra、随机图策略、单步 greedy 没有想象候选，0/0 代表不适用。完整频次及终止原因见 JSON。",
              "", "## 3. 分支处 utility 与噪声", "",
              r"采用 $r=(u_{(1)}-u_{(2)})/(\sqrt{2}\sigma)$，其中 utility 按全动作 L2 范数归一化。",
              "只统计实际到达、至少有两个合法出边的决策；表中每次重规划都有独立记录。",
              f"有噪声分支决策 {len(ratios)} 次；r 中位数 {number(analysis['noise_summary']['gap_ratio_median'])}，最大值 {number(analysis['noise_summary']['gap_ratio_max'])}，小于 1 的次数 {sum(value < 1 for value in ratios)}。",
              "r 很小时，前两条合法边之间的采样差异主要由噪声决定；这只是两动作比较，不等于整条路线成功概率。",
              "", "## 4. 候选的真实图连续性", "",
              f"独立重算候选数 {candidate_totals['candidates']}，连续合法 {candidate_totals['continuous_valid']}，预测到达 {candidate_totals['predicted_arrived']}，图审计到达 {candidate_totals['graph_arrived']}。",
              f"选中的候选数 {candidate_totals['selected_candidates']}，其中图审计到达 {candidate_totals['selected_graph_arrived']}；选中但虚假到达的候选 {len(selected_invalid)}。",
              "所有选择均先由模型完成，审计没有更改候选或选择。JSON 保存每条选中候选的动作及第一处非法转移。",
              "", "## 5. 解释边界", "",
              "本结果针对当前单场景、既定支持任务及冻结参数；源状态可行性拟合、汇点外推、长程想象和实际控制应分别解释。",
              "分支重复发生时，统计会重复计数；随机噪声仍不能保证候选预算内包含一条完整有效路线。",
              "此文件没有提出或实施基于本批任务结果的参数筛选。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

def display(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        if 0 < abs(value) < 10 ** (-digits):
            return f"{value:.4e}"
        return f"{value:.{digits}f}"
    return str(value)

def render_training_report(comparison):
    runs = comparison["runs"]
    focus = comparison["focus_contract"]
    bins = ("1", "2-4", "5-8", ">8")
    lines = ["# CML 训练覆盖与轮次对照", "", "## 1. 比较条件", "",
             "本报告只读取已经保存的实验。离线最短路标签用于诊断，没有反馈给模型或训练。",
             f"图完全一致：{comparison['identity']['all_graphs_identical']}；任务计划完全一致：{comparison['identity']['all_plans_identical']}。",
             "这是同场景开发诊断；不能据此声明跨场景泛化、多种子结论或正式视觉导航结果。", "",
             "| 运行 | 轮数 | 每轮转移 | 轨迹数 | 已覆盖边/总边 | 每边次数最小/最大 | 总消费转移 | 最后轮局部 MSE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in runs:
        c, t = run["coverage"], run["training"]
        visits = c["visits_per_edge_per_epoch"]
        lines.append(f"| {run['run_id']} | {t['epochs']} | {c['transitions_per_epoch']} | {c['nonempty_trajectories']} | {c['covered_edges']}/{c['edges']} | {visits['min']}/{visits['max']} | {t['updates_added']} | {display(t['epoch_prediction_mse'][-1])} |")
    lines += ["", "消费转移数是样本出现次数，不能在重复高级索引写入时解释为独立累加更新数。增加轮数与改善覆盖分别列出；若每轮预算不同，不能把全部差异仅归于采样方式。",
              "历史随机游走与均衡采样的比较同时改变覆盖、每轮消费量、轨迹切分和重复索引出现情况。不能把差异单独归因于某一种因素。以下单独核对各运行的重放序列和学习设置。", "",
              "| 运行 | 重复目标轨迹 | 重复动作轨迹 |", "| --- | ---: | ---: |"]
    for run in runs:
        lines.append(f"| {run['run_id']} | {run['coverage']['repeated_target_trajectories']} | {run['coverage']['repeated_action_trajectories']} |")
    lines += ["", "| 两次运行 | 重放序列完全一致 | 学习规则与初始化设置一致 | 轮数 |", "| --- | --- | --- | --- |"]
    for pair in comparison["training_controls"]:
        lines.append(f"| {pair['first_run']} → {pair['second_run']} | {pair['same_replay_trajectories']} | {pair['same_learner_settings']} | {pair['epochs']} |")
    lines += ["",
              "## 2. 全部任务与方法", "",
              "| 运行 | 方法 | 任务 ID | 起点/终点节点 | 端点支持 | 图可达 | 成功 | SPL | 环境步 | 图尝试/完成 | 终止原因 |",
              "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    actual_notes = []
    for run in runs:
        for task in run["tasks"]:
            s = task["summary"]
            lines.append(f"| {run['run_id']} | {task['method']} | {task['episode_id']} | {s['start_node']}/{s['goal_node']} | {s['endpoints_supported']} | {s['directed_graph_reachable']} | {display(s['metrics'].get('success'))} | {display(s['metrics'].get('spl'))} | {s['steps']} | {s['graph_hops_attempted']}/{s['graph_hops_completed']} | {s['end_reason']} |")
    lines += ["", f"## 3. 节点 {focus['source_node']} 到目标 {focus['goal_node']} 的候选", "",
              "下表为同一冻结模型、空动作历史下的静态选择。真实到达该位置时的历史、分支和选边单独保存在 comparison.json；没有到达不等于运行时做出了此选择。", "",
              "| 运行 | 边 | 后继 | W | V | 后继 Q 距目标 | Q+V 距目标 | 转移残差 | 每轮/本次训练累计次数 | 静态选择 | 分支 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for run in runs:
        critical = run[focus["result_key"]]
        for item in critical.get("candidate_values", []):
            lines.append(f"| {run['run_id']} | {item['edge_id']} | {item['target_node']} | {display(item['w_dot'])} | {display(item['v_dot'])} | {display(item['successor_q_goal_distance'])} | {display(item['predicted_q_plus_v_goal_distance'])} | {display(item['transition_residual_norm'])} | {item['visits_per_epoch']}/{item['visits_per_epoch'] * run['training']['epochs']} | {item['edge_id'] == critical['history_empty_selected_edge']} | {critical['history_empty_decision']['branch']} |")
        actual = critical.get("actual_runtime_decisions", [])
        if actual:
            selected = ", ".join(f"{x['episode_id']}: 边 {x['decision']['selected_edge']} ({x['decision']['cml_selection']['branch']})" for x in actual)
            actual_notes.append(f"- {run['run_id']} 真实记录：{selected}。")
        else:
            actual_notes.append(f"- {run['run_id']} 没有该状态—目标组合的真实决策记录。")
    lines += ["", *actual_notes, "", "## 4. 分支节点上的局部目标选择诊断", "",
              f"穷举共享图中出度至少为 2、目标可达且不同于当前节点的全部 {comparison['diagnostic_pairs']['count']} 对，没有抽样。分桶使用最少有向跳数；正确标签使用米制边代价的最短路，允许并列最优出边。",
              "每对都清空动作历史，只评估一次选边。该准确率不等于完整导航成功率。不可达选边的额外代价视为无穷，在 JSON 中记 null 并另计数；有限额外代价均值仅针对仍可达的选择。", "",
              "| 运行 | 目标跳数 | 对数 | CML 正确率 | W 最大正确率 | V 最大正确率 | CML 选入不可达数 | 有限额外代价均值 m |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in runs:
        methods = run["choice_diagnostics"]["methods"]
        for key in (*bins, "all"):
            m = methods["cml_history_empty"][key]
            lines.append(f"| {run['run_id']} | {key} | {m['pairs']} | {display(m['accuracy'])} | {display(methods['w_argmax'][key]['accuracy'])} | {display(methods['v_argmax'][key]['accuracy'])} | {m['unreachable_choices']} | {display(m['excess_cost_m_conditional_on_reachable_choice']['mean'])} |")
    lines += ["", "## 5. 所有边的最终转移残差", "",
              "残差为 ‖Q[:,目标]−Q[:,源]−V[:,边]‖₂，与按轨迹平均的训练 MSE 不是同一统计量。", "",
              "| 运行 | 边组 | 数量 | 均值 | 中位数 | P95 | 最大值 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for run in runs:
        for group, stats in run["residual_norms"].items():
            lines.append(f"| {run['run_id']} | {group} | {stats['count']} | {display(stats['mean'])} | {display(stats['median'])} | {display(stats['p95'])} | {display(stats['max'])} |")
    lines += ["", "## 6. 基准 Dijkstra 路线的每条边", "",
              "固定使用第一个输入运行中成功的 Dijkstra 路线。后续运行不重新选择便于比较的路线。每个单元格为每轮次数 / 本次 fit 累计消费次数。", ""]
    for route in comparison["reference_dijkstra_routes"]:
        lines += [f"### 任务 {route['episode_id']}（{len(route['selected_edges'])} 条边）", "",
                  "| 跳 | 边 | 源→目标 | " + " | ".join(run["run_id"] for run in runs) + " |",
                  "| ---: | ---: | --- | " + " | ".join("---:" for _ in runs) + " |"]
        for hop, edge in enumerate(route["selected_edges"]):
            rows = [next(x for x in run["reference_dijkstra_routes"] if x["task"] == route["task"])["edges"][hop] for run in runs]
            lines.append(f"| {hop} | {edge} | {rows[0]['source_node']}→{rows[0]['target_node']} | " + " | ".join(f"{x['visits_per_epoch']}/{x['consumed_visits_this_fit']}" for x in rows) + " |")
    lines += ["", "## 7. 数值证据与边界", "",
              "- comparison.json：配置、所有任务摘要与图决策、指标定义、来源文件哈希、源码哈希、完整比较结果。",
              "- diagnostic_pairs.jsonl：所有运行共用的状态—目标对、候选边、最短路标签和每个候选的图代价。",
              "- run_*/edge_diagnostics.json：每条边的最终残差、向量范数、训练频次。",
              "- run_*/choices.jsonl.gz：每个共享 pair_id 的全部候选 W/V 评分、静态选边、分支及相对最优代价；按 diagnostic_pairs.jsonl 中的 candidate_edges 顺序对齐。",
              "- loss 下降、残差降低与完整导航成功分别报告；不能仅凭其中一个指标推断其他指标。",
              "- 这里只比较给定运行和种子；端点覆盖、图支持范围、更多训练与记忆机制的影响仍需分别解释。", ""]
    return "\n".join(lines)


def write_full_goal_report(result: dict[str, Any], output: Path) -> None:
    """Render checked full-goal records without embedding one run's findings."""
    from collections import Counter

    tasks, totals, protocol = result["tasks"], result["totals"], result["planning_protocol"]
    candidates = totals["candidates"]
    cycles = totals.get("retrieved_state_cycle", 0)
    empty = totals.get("empty_affordance", 0)
    cycle_lengths = Counter()
    for task in tasks:
        cycle_lengths.update({int(key): value for key, value in task["cycle_lengths"].items()})
    date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    lines = ["# 无子目标 GCML：首次 no_plan 的逐候选复核", "",
             f"原始运行：`{result['run']}`；条件：`gcml_graph_affordance`。分析日期：{date}。", "",
             "本报告只读取历史规划与执行日志，没有新增训练、规划或仿真。距离与可完成性是事后分析标签。", "",
             "## 1. 真实执行与任务范围", "",
             f"共 {len(tasks)} 个任务。契约要求每个任务只有一次初始图决策、没有选中或执行任何图边。",
             "起点位置到图节点的局部连接动作及随后 STOP 单独计数，不能解释为执行了想象候选。", "",
             "| 任务 / episode | 起点→目标 | 起点连接动作数 | 总原语动作数 | 实际图边 | Dijkstra 实际图边 | Dijkstra 成功 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task in tasks:
        lines.append(f"| {task['task']} / {task['episode_id']} | {task['start_node']}→{task['goal_node']} | {task['connector_steps']} | {task['primitive_steps_including_stop']} | {task['selected_graph_edges']} | {task['dijkstra_executed_graph_hops']} | {task['dijkstra_success']} |")
    lines += ["", "## 2. 完整候选统计", "",
              "| 任务 | 回环 | 汇点 | 长度 min / mean / max | 不同节点序列数 | 最少有向边数 |",
              "|---|---:|---:|---|---:|---:|"]
    for task in tasks:
        count, length = task["counts"], task["lengths"]
        lines.append(f"| {task['task']} | {count.get('retrieved_state_cycle', 0)} | {count.get('empty_affordance', 0)} | {length['min']} / {length['mean']:.2f} / {length['max']} | {len(task['distinct_routes'])} | {task['minimum_graph_hops']} |")
    lines += ["", f"共 {candidates} 条候选：回环 {cycles}（{cycles / candidates:.1%}），汇点 {empty}（{empty / candidates:.1%}）。",
              f"全部 {totals['legal_transitions']} 次想象转移连续合法，逐步 latent 关联与边目标一致。",
              f"最长候选 {max(task['lengths']['max'] for task in tasks)} 条边；规划深度预算为 {protocol.get('horizon', protocol.get('max_depth', '见 planning_protocol'))}。",
              "候选在回环后终止，没有在候选内部回溯换分支；汇点仅表示经验图没有出边，不代表物理环境没有出口。", "",
              "| 回环边数 | 候选数 |", "|---:|---:|"]
    lines.extend(f"| {length} | {count} |" for length, count in sorted(cycle_lengths.items()))
    reachable_cycles = totals.get("cycle_terminal_globally_reachable", 0)
    alternative_cycles = totals.get("cycle_last_has_simple_completion_alternative", 0)
    lines += ["", f"{reachable_cycles} 条回环的终止节点在完整经验图中仍可到达目标；{alternative_cycles} 条在最后选边时仍存在避开此前节点的完成路径。", "",
              "| 任务 | 汇点与候选数量 |", "|---|---|"]
    for task in tasks:
        sinks = "; ".join(f"节点 {row['node']}: {row['count']}" for row in task["terminal_node_counts"] if row["termination"] == "empty_affordance")
        lines.append(f"| {task['task']} | {sinks} |")
    lines += ["", "## 3. 首次失去简单完成路径的实例", "",
              "以下按记录顺序选取最多三个不同失败类别的实例；完整候选都保存在 JSON，没有据结果重新筛选任务。", ""]
    selected_kinds = set()
    for task in tasks:
        for candidate in task["candidates"]:
            failure = candidate["first_lost_simple_completion"]
            key = (failure["kind"], failure["utility_top_has_simple_completion"])
            if key in selected_kinds or len(selected_kinds) >= 3:
                continue
            selected_kinds.add(key)
            lines += [f"### {task['task']} / candidate {candidate['id']}", "",
                      f"节点序列：`{' → '.join(map(str, candidate['nodes']))}`。",
                      f"首次失去简单完成路径：节点 {failure['node']}，选中边 {failure['action']}→{failure['target']}，类别 `{failure['kind']}`。",
                      f"仍可简单完成的出边：`{failure['simple_completion_edges']}`。", "",
                      "| 边 → 目标 | 归一化 utility | 噪声 | 相加分数 | 选中 |",
                      "|---|---:|---:|---:|---|"]
            for score in failure["saved_scores"]:
                lines.append(f"| {score['edge']}→{score['target']} | {score['normalized_utility']:.6f} | {score['noise']:.6f} | {score['normalized_utility_plus_noise']:.6f} | {score['selected']} |")
            lines.append("")
    lines += ["## 4. 评分、噪声与 no_plan", "",
              r"每步分数为 $\tilde u_a+\epsilon_a$，$u=W(Q_g-z)$，$\tilde u=u/\|u\|_2$；范数覆盖全部动作。",
              f"本次保存的噪声标准差为 {protocol['noise_scale']}。经验图门控只提供合法出边，不提供到目标的距离、死路标签或已访问节点屏蔽。",
              f"{totals['branch_steps']} 次多出边选择中，噪声改变 argmax 共 {totals['noise_changes_argmax']} 次。",
              f"utility 最高边为代价最优出边 {totals['utility_top_cost_optimal']}/{totals['reachable_branch_steps']} 次；实际选择为代价最优出边 {totals['selected_cost_optimal']}/{totals['reachable_branch_steps']} 次。",
              f"首次失去简单完成路径时，utility 最高边仍可完成的候选共 {totals['first_trap_utility_top_viable']} 条，最高边本身已不可完成的候选共 {totals['first_trap_utility_top_not_viable']} 条。",
              "这些是重复候选内部的条件统计，不是独立任务或零噪声重跑成功率；非最短路选择也可能完成任务。",
              "规划器只接受预测完整到达的候选。此适配器已核对所有候选均未预测到达，因此没有执行合法部分前缀。", "",
              "## 5. 与 G 和状态漂移的关系", "",
              "本条件使用经验图门控，绕过学习到的 G。因此不能用 G 假阴性解释这些失败。",
              f"逐步关联错误为 0，最大关联距离 {result['association']['max_nearest_distance']:.8f}，最小相对间隔 {result['association']['min_relative_margin']:.8f}。",
              "图可达性、绕行与简单完成性均只用于分析，不反馈给原策略。本报告没有新增任何子目标或噪声消融实验。", "",
              f"全部 {candidates} 条候选：[diagnostics.json]({output / 'diagnostics.json'})。",
              f"全部不同节点路径：[candidate_routes.md]({output / 'candidate_routes.md'})。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    routes = [f"# 全部 {candidates} 条候选的节点路径索引", "",
              "相同节点序列合并，候选 ID 从 0 开始；保存的各步数值可能不同。", ""]
    for task in tasks:
        routes += [f"## {task['task']} / episode {task['episode_id']}", "",
                   "| 候选 ID | 数量 | 终止 | 节点路径 |", "|---|---:|---|---|"]
        by_id = {row["id"]: row for row in task["candidates"]}
        for route in task["distinct_routes"]:
            ids = ", ".join(str(value) for value in route["candidate_ids"])
            termination = by_id[route["candidate_ids"][0]]["termination"]
            nodes = " → ".join(map(str, route["nodes"]))
            routes.append(f"| {ids} | {route['count']} | {termination} | {nodes} |")
        routes.append("")
    (output / "candidate_routes.md").write_text("\n".join(routes), encoding="utf-8")

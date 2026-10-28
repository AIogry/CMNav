"""Historical dimension and mechanism Markdown templates from validated analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from cmnav.analysis.report_contracts import NOISE_CONDITIONS
from cmnav.analysis.mechanism_comparison import successful
from cmnav.reporting.labels import GCML_SHORT_LABELS as SHORT_LABELS, GCML_LABELS, MECHANISM_LABELS as LABELS


def number(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def dimension_markdown(run: Path, data: dict[str, Any]) -> str:
    dimensions, methods = data["dimensions"], data["methods"]
    protocol = data["protocol"]
    control = data["renderer_control"]
    settings, planning = data["shared_settings"], data["planning_settings"]
    task_count = settings["task_count"]
    training = data["results"][str(dimensions[0])]["training"]
    lines = ["# GCML 维度对照：完整目标导航", "", f"运行目录：`{run}`", "",
             "## 1. 本轮能回答什么", "",
             f"本轮比较维度 {dimensions}，使用同一份 {settings['node_count']} 节点、{settings['edge_count']} 条有向边的经验图与冻结经验批次。"
             f"各档 seed={settings['seed']}，训练 {settings['epochs']} 轮；每轮 {settings['trajectory_batches_per_epoch']} 批、{settings['transitions_per_epoch']} 次转移。"
             f"全部 {task_count} 个冻结任务都纳入比较；参数、预算和任务身份由保存配置与清单核对。", "",
             f"每次规划 {planning['gcml_candidates']} 条候选、上限 {planning['gcml_horizon']} 条边，噪声标准差 {planning['gcml_noise_scale']}；"
             f"G 阈值 {planning['gcml_affordance_threshold']}，关联半径 L2={planning['gcml_association_radius']}。"
             f"原始训练与评测配置保留在 comparison.json，各任务使用其记录的执行协议。", "",
             "| 维度 | 方法 | 成功数 | no_plan 任务数 | 平均实际步数 | 完成图边总数 |",
             "| ---: | --- | ---: | ---: | ---: | ---: |"]
    failures = []
    for dim in dimensions:
        rows = data["results"][str(dim)]["tasks"]
        for method in methods:
            group = [row for row in rows if row["method"] == method]
            success = sum(bool(row["metrics"]["success"]) for row in group)
            no_plan = sum(row["end_reason"] == "no_plan" for row in group)
            failures.append((dim, method, success, no_plan))
            lines.append(f"| {dim} | {SHORT_LABELS[method]} | {success}/{len(group)} | {no_plan} | "
                         f"{np.mean([row['steps'] for row in group]):.1f} | {sum(row['graph_hops_completed'] for row in group)} |")
    imagined = [row for row in failures if row[1] in {"gcml", "gcml_graph_affordance"}]
    if imagined and all(row[2] == 0 for row in imagined):
        lines += ["", f"本轮观测：声明维度下的想象规划方法均未完成任务；当前比较没有解决这 {task_count} 个任务的完整目标导航问题。候选生成变化还需结合第 6 节的到达、合法性和终止原因判断。"]
    else:
        lines += ["", f"本轮成功数见上表；任何改善仅描述当前图、{task_count} 个任务与每维度一个训练种子，尚不能推断更高维度普遍更好。"]
    lines += ["", "`no_plan` 时较少的实际环境步数不能当作效率收益。若完成图边为零，动作只用于起点连接与 STOP。", "",
              "## 2. 论文依据、对照与运行条件", "",
              f"- [GCML 正文]({protocol['paper']})；维度取值依据冻结的 sweep protocol，完整引用另见算法 reference。",
              f"- [官方代码]({data['algorithm_reference']['repository']})；[固定版本 notebook]({data['algorithm_reference']['source_url']})。"]
    for dim in dimensions:
        lines.append(f"- {dim} 维取值依据：`{protocol.get('dimension_rationale', {}).get(str(dim), '见 protocol.json')}`。")
    lines += ["- 这些是论文中的实验设置参考，没有把它们解释成适用于任何节点数或路径长度的维度定律；本实现仍是已声明的 PointNav 适配。",
              f"- 本次 renderer device={data['renderer_device']}；使用原 checkpoint 的 renderer control 已完成。"
              f"逐任务完整比较 {control['matching_tasks']}/{control['tasks']} 一致，全部一致={control['all_match']}。",
              "- renderer control 同时比较实际动作、状态、指标、去时间规划日志，以及每条候选的动作/终止/选择；差异写入原始 control_comparisons.json，未据此调参。"]
    for dim in dimensions:
        baseline = data["results"][str(dim)]["training_reproduction"]
        if baseline.get("performed"):
            lines.append(f"- {dim} 维重训与其声明原 checkpoint 的 Q/V/W/G 完全一致：{baseline.get('all_tensors_exact')}。")
            if baseline.get("tensors"):
                lines.append("  各张量最大绝对差：" + "，".join(f"{key}={number(value['max_absolute_delta'])}" for key, value in baseline["tensors"].items()) + "。")
    if control["summary"].get("methods", {}).get("dijkstra"):
        ref = control["summary"]["methods"]["dijkstra"]
        lines.append(f"- renderer control 的 Dijkstra 成功 {round(ref['success'] * ref['episodes'])}/{ref['episodes']}；路径图及任务没有因增维改变。")
    lines += ["", "## 3. 训练与读出", "",
              "下表 G 指标使用 float32 的 GQ 读出；监督源节点按原始训练 trajectories 的 source 去重，汇点按经验图出度为零确定。"
              "汇点没有本轮训练监督；G 指标计算不会写回模型。Q 的秩与条件数由 float64 SVD 计算。", "",
              "| 维度 | source 数 | source FP/FN | source RMSE | sink 数/FP | Q 秩 | Q 条件数 | 转移 L2 最大误差 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dim in dimensions:
        item = data["results"][str(dim)]
        g, q = item["readout"], item["Q_svd"]
        source, sink = g["supervised_sources"], g["graph_sinks"]
        lines.append(f"| {dim} | {source['states']} | {source['false_positive']}/{source['false_negative']} | {number(source['rmse'])} | "
                     f"{sink['states']}/{sink['false_positive']} | {q['rank']}/{q['columns']} | {number(q['condition_number'])} | "
                     f"{number(item['training']['final_diagnostics']['prediction_error_l2_max'])} |")
    lines += ["", "| 维度 | W 实际步长最小值 | W 封顶批数 | G 实际步长最小值 | G 封顶批数 | 训练秒数 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dim in dimensions:
        item = data["results"][str(dim)]
        rates = item["readout_rates"]
        lines.append(f"| {dim} | {number(rates['W']['minimum'])} | {rates['W']['capped_batches']} | "
                     f"{number(rates['G']['minimum'])} | {rates['G']['capped_batches']} | {item['training']['elapsed_seconds']:.1f} |")
    lines += ["", f"W/G requested rates 分别为 {training['requested_inverse_learning_rate']}/{training['requested_affordance_learning_rate']}；"
              f"保存的步长规则为 `{training['readout_step_rule']}`，safety={training['readout_step_safety']}。"
              "增维改变范数和可能触发的有效步长，训练预算相同不意味着优化动力学完全相同。", "",
              "上述秒数是保存的训练墙钟耗时，可能包含机器负载与实现因素，不能直接作为独占硬件下的维度复杂度基准。", "",
              f"![训练曲线]({data['figures'][0]})", "", "## 4. 面向远目标的 W 方向判断", "",
              f"只在 {task_count} 条冻结的成本最优参考路径上做事后诊断，目标始终为各任务最终目标。"
              f"先计算 W(Q_goal−Q_state)，按全部 {settings['edge_count']} 个动作作 L2 归一化，再在真实出边中选最大分。"
              "任何满足边代价+后继最短代价=当前最短代价的出边都算正确，允许等成本路线。", "",
              "gap=最佳正确出边分−最佳非最优出边分；没有非最优竞争边时 gap 不定义，未以零补入均值。"
              f"gap/σ 比较评分差和单动作噪声，σ={planning['gcml_noise_scale']}；两动作独立噪声之差的标准差为 √2σ。σ=0 时比值不定义，另存 JSON 为 null。"
              f"这些量不是完整 {planning['gcml_candidates']} 条候选的成功概率。", "",
              "| 维度 | 所有参考节点正确 | 多出边节点正确 | gap/σ 中位数 | gap/σ 平均值 | gap<0 节点数 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dim in dimensions:
        values = data["results"][str(dim)]["long_goal_W"]["summary"]
        lines.append(f"| {dim} | {values['correct']}/{values['states']} | {values['branching_correct']}/{values['branching_states']} | "
                     f"{number(values['gap_over_noise_sigma']['median'])} | {number(values['gap_over_noise_sigma']['mean'])} | {values['negative_gap_states']} |")
    lines += ["", "参考路径上的节点样本有结构关联；不是额外独立测试任务，也没有给 planner 提供最短路标签。", "",
              f"![维度比较]({data['figures'][1]})", "", f"## 5. 全部 {task_count} 个任务", "",
              "| 维度 | 任务/episode | 方法 | 成功 | 实际步数 | 完成/尝试图边 | 终止原因 |",
              "| ---: | --- | --- | ---: | ---: | ---: | --- |"]
    for dim in dimensions:
        for row in data["results"][str(dim)]["tasks"]:
            lines.append(f"| {dim} | {row['task_index']}/{row['episode_id']} | {SHORT_LABELS[row['method']]} | "
                         f"{int(bool(row['metrics']['success']))} | {row['steps']} | {row['graph_hops_completed']}/{row['graph_hops_attempted']} | {row['end_reason']} |")
    lines += ["", "## 6. 每任务候选诊断", "",
              "预测到达不保证图审计到达，因此同时列出非法候选与虚假到达。下表统计全部实际决策的候选，"
              "同一维度不同方法的决策数可能不同；Greedy 不生成候选，零计数没有规划失败含义。关联失败终止列仅统计 state_unassociated 终止的候选，并不等同于学得 G 的所有关联失败步骤。", "",
              "| 维度 | 任务 | 方法 | 候选 | 预测到达 | 非法 | 虚到达 | 回环 | 空集 | 关联失败终止 | 耗尽 |",
              "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dim in dimensions:
        for row in data["results"][str(dim)]["tasks"]:
            if row["method"] not in {"gcml", "gcml_graph_affordance"}:
                continue
            p = row["gcml_planning"]
            t = p["candidate_terminations"]
            lines.append(f"| {dim} | {row['task_index']} | {SHORT_LABELS[row['method']]} | {p['candidates']} | "
                         f"{p['predicted_arrived_candidates']} | {p['invalid_candidates']} | {p['false_predicted_arrival_candidates']} | "
                         f"{t.get('retrieved_state_cycle', 0)} | {t.get('empty_affordance', 0)} | {t.get('state_unassociated', 0)} | {t.get('horizon_exhausted', 0)} |")
    lines += ["", "## 7. 首次规划的候选长度与多样性", "",
              "| 维度 | 任务 | 方法 | 候选数 | 长度 min/mean/max | 不同动作序列数 | 首条执行边 |",
              "| ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for dim in dimensions:
        for row in data["results"][str(dim)]["tasks"]:
            p = row["first_candidates"]
            if not p["applicable"]:
                continue
            lengths = p["lengths"]
            edge = (p["first_graph_decision"] or {}).get("selected_edge")
            lines.append(f"| {dim} | {row['task_index']} | {SHORT_LABELS[row['method']]} | {lengths['count']} | "
                         f"{number(lengths['min'])}/{number(lengths['mean'])}/{number(lengths['max'])} | {p['distinct_action_sequences']} | {edge if edge is not None else '无'} |")
    lines += ["", "## 8. 解释边界与产物", "",
              "- 每维度只有一个训练种子，同 seed 的不同维度也不代表 Q/V/W/G 是严格嵌套的同一初始化；尚无多种子置信区间。",
              "- 当前是单场景、真值节点关联、冻结经验图与特权局部执行器的开发诊断，不能作为视觉导航或跨场景泛化结论。",
              "- 扩维仍保留原始循环终止、完整候选才允许执行等规则。经验图辅助提供合法出边，并不替模型选择通向最终目标的分支。",
              "- 源节点/汇点读出分别统计；报告只读取声明的 checkpoint，不进行补监督、门控调整或重新训练。",
              "- comparison.json 保存全部数值、每条参考路线每节点 W 分数、逐轮步长、Q 奇异值与输入指纹。",
              "- 原任务目录保留完整动作、位姿、视频预览及候选逐步日志；本脚本只读分析这些产物，没有新训练或导航。", ""]
    suffix_analysis = run / "replanning_suffix_analysis.md"
    if suffix_analysis.exists():
        lines += ["## 9. 独立后续诊断", "",
                  f"该运行另有[重规划后缀分析]({suffix_analysis})。其中的样本、条件和解释以该独立记录为准。", ""]
    return "\n".join(lines)


def exploratory_markdown(result: dict[str, Any] | None, run: str) -> list[str]:
    if result is None:
        return []
    baseline, treatment = result["baseline"], result["treatment"]
    protocol = result["protocol"]
    lines = ["", "## 6. 探索性补充：噪声 0.02 的旧计划后缀", "",
             "**这部分是在查看主实验开发任务结果后追加的 post-hoc 探索，不属于原定主实验矩阵。**"
             "选择该噪声，是因为追加决策时已完成的 0.05 缓存运行尚未发生回退选择，而 0.02 基线存在已经走过图边、随后 no_plan 的案例。"
             "下面保留全部五任务和全部指定规划种子，不仅报告触发追加实验的案例。",
             f"独立清单中的 {result['completed_jobs']}/{result['planned_jobs']} 个追加 job 已完成；复用主实验 `{protocol['baseline_condition']}` 作为基线。"
             "报告器已核对同 checkpoint 文件、epoch、方法、种子和其他覆盖参数，唯一设置变化为 `cache_mode: none → suffix_on_no_plan`。"
             "本补充没有增加训练，不改变上面的主实验图与主条件汇总。", "",
             "| 规划种子 | 无缓存成功/5 | 有缓存成功/5 | 无缓存任务 no_plan | 有缓存任务 no_plan |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    before = {item["planning_seed"]: item for item in baseline["per_seed"]}
    after = {item["planning_seed"]: item for item in treatment["per_seed"]}
    for seed in protocol["planning_seeds"]:
        left, right = before[seed], after[seed]
        lines.append(f"| {seed} | {left['successes']}/5 | {right['successes']}/5 | {left['no_plan_tasks']} | {right['no_plan_tasks']} |")
    pair = result["paired_contrast"]
    lines += ["", f"匹配 {pair['matched_seed_task_repetitions']} 次种子×任务执行：失败→成功 {pair['improved_repetitions']} 次、"
              f"成功→失败 {pair['worsened_repetitions']} 次、成功状态不变 {pair['unchanged_repetitions']} 次。"
              "它们仍然只有五个固定开发任务、一个训练种子，不构成 15 个独立测试样本。", "",
              "| 种子／任务 | 无缓存：结局／完成图跳 | 有缓存：结局／完成图跳 | 基线已行走后 no_plan | 存入计划 | 回退选择／完成 | 非法回退 |",
              "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for item in result["task_pairs"]:
        left, right = item["baseline"], item["treatment"]
        counts = right["cache"]["counts"]
        lines.append(f"| {item['planning_seed']}／{item['task_index']:04d} | {left['end_reason']}／{left['graph_hops_completed']} | "
                     f"{right['end_reason']}／{right['graph_hops_completed']} | {'是' if item['baseline_failed_after_completed_graph_hops'] else '否'} | "
                     f"{counts['plans_stored']} | {counts['fallback_actions_selected']}/{counts['fallback_actions_completed']} | {counts['invalid_fallback_actions']} |")
    lines += ["", "| 条件 | 候选数 | 非法候选 | 虚假到达 | 回环 | 空门控 | 深度耗尽 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, group in (("无缓存 0.02", baseline), ("有缓存 0.02", treatment)):
        counts, term = group["planning_counts"], group["candidate_terminations"]
        lines.append(f"| {label} | {counts['candidates']} | {counts['invalid_candidates']} | {counts['false_predicted_arrival_candidates']} | "
                     f"{term.get('retrieved_state_cycle', 0)} | {term.get('empty_affordance', 0)} | {term.get('horizon_exhausted', 0)} |")
    cache = treatment["cache"]
    counts = cache["counts"]
    lines += ["", f"缓存共存入 {counts['plans_stored']} 次，回退尝试 {counts['fallback_attempts']} 次，未命中 {counts['fallback_cache_misses']} 次，"
              f"回退选择 {counts['fallback_actions_selected']} 条边、完成 {counts['fallback_actions_completed']} 条边。"
              f"有计划存入的执行 {cache['task_repetitions_with_plan_stored']}/{treatment['seed_task_repetitions']} 次，实际使用回退的执行 "
              f"{cache['task_repetitions_with_fallback_used']}/{treatment['seed_task_repetitions']} 次。",
              f"缓存失效原因：`{json.dumps(cache['invalidation_reasons'], ensure_ascii=False, sort_keys=True)}`。",
              "每步仍先完成原预算的新规划，失败才使用后缀，不能据此声称节省了该次搜索计算。缓存改变后续状态和决策次数，因此同种子配对也不意味着之后的全部搜索事件相同。",
              "追加条件的选择依赖已有开发结果；任何改善都应作为后续独立任务验证的线索，不作为未经选择偏差影响的确认性结论。",
              f"补充协议：[exploratory_cache_002_protocol.json]({Path(run) / 'exploratory_cache_002_protocol.json'})；"
              f"独立清单：[exploratory_jobs.json]({Path(run) / 'exploratory_jobs.json'})。"]
    return lines


def format_value(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}g}"


def control_audit_markdown(study: dict[str, Any]) -> list[str]:
    """Summarize saved audit findings without reexecuting a model or episode."""
    lines: list[str] = []
    historical = study["control_audits"].get("historical_gcml_cuda_comparison")
    if historical is not None:
        data = historical["data"]
        cpu = data["aggregate_cpu_excluding_times"]
        gpu = data["aggregate_gpu_excluding_times"]
        totals, conclusion, scope = data["totals"], data["conclusion"], data["scope"]
        checkpoint = data["checkpoint"]
        checkpoint_equal = (checkpoint.get("recorded_file_sha256_equal") is True
                            and checkpoint.get("recorded_file_sha256_matches_actual") is True
                            and checkpoint.get("recorded_parameter_tensor_sha256_equal") is True)
        exact_labels = (("五任务结局", conclusion["same_task_outcomes"]),
                        ("完整物理执行日志", conclusion["same_physical_execution"]),
                        ("离散规划决策", conclusion["same_planning_discrete_decisions"]))
        lines += ["", "### 同一历史 checkpoint 的 CPU/CUDA 推理对照", "",
                  f"这一控制使用同一份历史 5000 维、epoch 100 checkpoint；文件与参数哈希核验：{'一致' if checkpoint_equal else '未全部一致'}。"
                  f"规划基础种子 {scope['planning_base_seed']}，噪声 {scope['noise_scale']}，调度 `{scope['noise_schedule']}`。",
                  "它与上面的 CPU/GPU 训练参数差异表回答不同问题：这里固定历史 checkpoint，对比保存的逐步推理和物理执行日志。", "",
                  "| 指标 | 历史 CPU | 同 checkpoint CUDA |", "| --- | ---: | ---: |",
                  f"| 成功任务 | {round(cpu['success'] * cpu['episodes'])}/{cpu['episodes']} | {round(gpu['success'] * gpu['episodes'])}/{gpu['episodes']} |",
                  f"| no_plan 任务 | {cpu['end_reasons'].get('no_plan', 0)} | {gpu['end_reasons'].get('no_plan', 0)} |",
                  f"| 候选数 | {cpu['gcml_planning']['candidates']} | {gpu['gcml_planning']['candidates']} |",
                  f"| 想象转移数 | {cpu['gcml_planning']['model_transitions']} | {gpu['gcml_planning']['model_transitions']} |", "",
                  "；".join(f"{label}：{'一致' if exact else '存在差异'}" for label, exact in exact_labels) + "。",
                  f"候选动作、终止、选择、成本及独立图审计一致 {totals['candidate_key_exact']}/{totals['candidates']}；"
                  f"想象步骤的动作、候选边、解码节点、关联标志及噪声一致 {totals['imagined_step_discrete_and_noise_exact']}/{totals['imagined_steps']}。"
                  f"物理原语动作共 {totals['executed_primitive_actions']} 次。", "",
                  "连续值仍存在浮点差异，以下是保存日志中逐元素比较的最大绝对差；不作逐位数值一致声明。", "",
                  "| 连续字段 | 不完全相等的值数 | 最大绝对差 |", "| --- | ---: | ---: |"]
        for key, values in data["candidate_float_differences"].items():
            field = key.removeprefix("candidate_records[].candidate.steps[].")
            lines.append(f"| `{field}` | {values['differing_value_count']} | {format_value(values['max_absolute_difference'], 6)} |")
        lines += ["", "这项结果仅覆盖该历史 checkpoint、五任务和 legacy 噪声调度的这个种子；不能推广为新 GPU 训练 checkpoint、完整索引噪声表、其他规划种子或其他机制条件的数值等价保证。",
                  f"依据：[已保存的 CPU/CUDA 日志审计]({historical['json_path']})；JSON 与 Markdown 均计入本报告输入 SHA256。"]
    remaining = study["control_audits"].get("remaining_historical_controls")
    if remaining is not None:
        data = remaining["data"]
        conclusion = data["conclusion"]
        task_count = sum(len(item["tasks"]) for item in data["comparisons"].values())
        successes = "、".join(f"`{method}` {count}/5" for method, count in conclusion["old_success_counts_preserved"].items())
        exact = (conclusion["all_15_task_summaries_exact"] and conclusion["all_15_execution_records_exact"]
                 and conclusion["all_graph_decisions_exact"] and conclusion["all_planning_and_candidate_nonfloat_fields_exact"])
        lines += ["", f"其余三个历史控制的 {task_count} 项任务结果、完整物理执行与离散规划决策{'全部一致' if exact else '存在差异'}，"
                  f"保留的成功数为 {successes}；GCML 连续分数仍有浮点差异，Dijkstra 控制主要核对任务、图及物理执行条件。"
                  f"这一结论同样限于审计列明的历史条件，详见[其余历史控制审计]({remaining['json_path']})，其 JSON/Markdown 已计入输入 SHA256。"]
    greedy = study["control_audits"].get("greedy_checkpoint_log_comparison")
    if greedy is not None:
        data = greedy["data"]
        detail = data["task2_epoch80_vs_epoch100"]
        outcomes = data["task_outcomes"]
        before_task = next(item for item in outcomes["new_gpu_epoch80"] if item["episode_id"] == detail["episode_id"])
        after_task = next(item for item in outcomes["new_gpu_epoch100"] if item["episode_id"] == detail["episode_id"])
        mse = detail["saved_full_checkpoint_inverse_mse"]
        cycle = detail["epoch80_cycle"]
        escape = detail["epoch100_node39_escape"]
        divergence = detail["first_divergence"]
        lines += ["", "### Epoch 80→100：整体 W 误差趋缓仍可能改变局部排序", "",
                  "以下来自已有 greedy 日志，使用真实当前出边和完整目标；没有重新评分或运行新轨迹。", "",
                  "| Checkpoint | 成功任务/5 | task 0002／66742 的动作数 | 任务结局 |",
                  "| --- | ---: | ---: | --- |"]
        for key, label in (("old_cpu_epoch100", "历史 CPU epoch100"), ("new_gpu_epoch80", "本次 GPU epoch80"),
                           ("new_gpu_epoch100", "本次 GPU epoch100")):
            task = next(item for item in outcomes[key] if item["episode_id"] == detail["episode_id"])
            count = sum(successful(item["success"]) for item in outcomes[key])
            lines.append(f"| {label} | {count}/{len(outcomes[key])} | {task['steps']} | {task['end_reason']} |")
        lines += ["", f"冻结 checkpoint 的全图边 `raw.inverse_mse` 从 {format_value(mse['80']['value'], 10)} 降至 "
                  f"{format_value(mse['100']['value'], 10)}，相对下降 **{format_value(detail['inverse_mse_relative_decrease_percent'], 5)}%**。"
                  "这个量不同于训练过程中的逐批均值；它的变化很小，但任务相关的局部动作排序仍然变化。", "",
                  f"第一次分岔发生在第 {divergence['decision_index'] + 1} 次决策、节点 {divergence['epoch80']['current_node']}；"
                  f"两条路线都在 hop {divergence['reconvergence_hop_both']} 回到节点 {divergence['reconvergence_node']}，因此不能把最早分岔直接当作失败原因。"
                  "持续导致预算耗尽的是后续回环。", "",
                  "| Checkpoint／当前节点 | 候选边及 W | 选中边 |",
                  "| --- | --- | --- |"]
        for label, decision in (("epoch80", cycle["node39_first"]), ("epoch80", cycle["node40_first"]),
                                ("epoch100", escape)):
            candidates = "；".join(f"ID {item['edge_id']}（{item['source_node']}→{item['target_node']}）：{format_value(item['W'], 8)}"
                                   for item in decision["candidate_scores"])
            source, target = decision["selected_edge_nodes"]
            lines.append(f"| {label}／{decision['current_node']} | {candidates} | ID {decision['selected_edge_id']}（{source}→{target}） |")
        selected_counts = "；".join(f"边 {item['edge_id']}（{item['edge_nodes'][0]}→{item['edge_nodes'][1]}）选择 {item['count']} 次"
                                    for item in cycle["selected_action_counts"])
        lines += ["", f"Epoch80 的 hop {cycle['first_hop']}–{cycle['last_hop']}：{selected_counts}。"
                  f"记录了 {cycle['alternating_selection_pairs']} 对交替选择，不能等同于全部物理完成的回环圈数；"
                  f"该任务总计尝试 {cycle['total_task_graph_hops_attempted']} 次图跳、完成 {cycle['total_task_graph_hops_completed']} 次。",
                  f"Epoch100 在节点 {escape['current_node']} 改选 ID {escape['selected_edge_id']}，胜出分差仅 "
                  f"**{format_value(escape['top1_minus_top2'], 8)}**，随后任务以 {after_task['steps']} 个环境动作成功；"
                  f"epoch80 在 {before_task['steps']} 步预算处终止。", "",
                  "这说明整体 W loss 趋缓不足以证明两个 checkpoint 的导航行为相同；该单例也不能确定一般性的早停阈值。"]
        comparisons = data["epoch100_old_cpu_vs_new_gpu"]
        action_matches = sum(item["all_primitive_actions_equal"] for item in comparisons)
        decision_matches = sum(item["all_decision_state_goal_edge_equal"] for item in comparisons)
        maximum_w_difference = max(item["max_absolute_saved_W_difference_on_aligned_routes"] for item in comparisons)
        lines += ["", f"另一个已有日志对照：本次 GPU epoch100 与历史 CPU epoch100 的物理动作序列一致 {action_matches}/{len(comparisons)} 个任务，"
                  f"逐决策 current/goal/edge 序列一致 {decision_matches}/{len(comparisons)} 个任务；对齐路线保存 W 值的最大绝对差为 {format_value(maximum_w_difference, 8)}。"
                  "这是分别训练得到的模型之间的行为对照，参数及分数不因此相同；也不能用它替代上节的同 checkpoint 设备控制。",
                  f"依据：[greedy checkpoint 日志审计]({greedy['json_path']})。节点与边端点经过冻结 graph.json 核对；审计、原日志、图和 checkpoint 诊断均记录 SHA256。"]
    return lines


def mechanism_markdown(study: dict[str, Any], groups: list[dict[str, Any]], paired: list[dict[str, Any]], output: Path,
             exploratory: dict[str, Any] | None = None) -> str:
    protocol = study["protocol"]
    final_epoch = protocol["epochs"]
    lookup = {(group["condition"]["name"], group["condition"]["epoch"]): group for group in groups}
    final = [lookup[name, final_epoch] for name in protocol["final_conditions"]]
    seeds = protocol["planning_seeds"]
    baseline_seed = seeds[0]
    completion_text = (f"主清单 {study['primary_completed_jobs']}/{study['primary_planned_jobs']} 个 job 已完成；"
                       + (f"观察后追加清单 {study['exploratory_completed_jobs']}/{study['exploratory_planned_jobs']} 个 job 已完成。"
                          if exploratory is not None else "没有追加的探索清单。"))
    lines = ["# GCML GPU 训练与推理机制对照", "", f"运行目录：`{study['run']}`。",
             completion_text + "所有已安排 job 完成后生成本报告；模型参数在每次任务中冻结。", "",
             f"**{protocol['dimension']} 维；一个训练种子 {protocol['training_seed']}；{final_epoch} epochs；同场景的五个固定开发任务。**",
             f"规划种子为 {', '.join(map(str, seeds))}。每个条件最多 15 次执行来自同五个任务的随机重复，不能视为 15 个独立任务或跨场景泛化样本。",
             "本轮全部面向最终目标，**没有教师子目标**。G 的最小二乘读出使用全部经验图节点标签，包括 13 个图中无出边节点，属于明确增加监督的诊断。", "",
             "## 1. 训练与 checkpoint", "",
             f"训练设备：`{study['training'].get('device')}`；评估设备：`{', '.join(study['evaluation_devices'])}`。",
             "原始 G 的训练误差、冻结 checkpoint 误差、额外拟合 G 的误差分别记录。逐批训练均值不是冻结模型的全图误差。", "",
             f"![训练与拟合诊断]({output / 'training_curves.png'})", "",
             "| Epoch | 原 Q/V 全边 MSE | 原 W 全边 MSE | 原 G 全节点 FP/FN | 原 G 汇点 FP | 最小改变量 G FP/FN | 最小范数 G FP/FN | LS rank / condition |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for point in study["checkpoints"]:
        raw = point["diagnostics"]["raw"]
        variants, solve = point["readout"]["variants"], point["readout"]["solve"]
        g0, gd, gn = (variants[name]["all_states"] for name in ("raw", "minimum_change", "minimum_norm"))
        lines.append(f"| {point['epoch']} | {format_value(raw.get('prediction_mse'))} | {format_value(raw.get('inverse_mse'))} | "
                     f"{g0['fp']}/{g0['fn']} | {variants['raw']['sinks']['fp']} | {gd['fp']}/{gd['fn']} | {gn['fp']}/{gn['fn']} | "
                     f"{solve.get('rank', '—')} / {format_value(solve.get('condition', solve.get('retained_condition')))} |")
    lines += ["", "G 的最小改变量解保留原 G 在未受 Q 监督方向上的分量；最小范数解同时清除这些分量。已知节点的拟合程度由上表和 comparison.json 记录；即使两者在这些节点上都准确，也不保证对偏离码本的连续想象状态给出相同门控。", "",
              f"以下 checkpoint 曲线统一使用规划种子 {baseline_seed}；没有挑选成绩最好的 epoch 作为唯一结果。", "",
              "| Epoch | 原 G 想象成功/5 | 拟合 G 想象成功/5 | 当前经验图 greedy 成功/5 |",
              "| ---: | ---: | ---: | ---: |"]
    for epoch in protocol["checkpoints"]:
        values = [next(block["successes"] for block in lookup[name, epoch]["per_seed"] if block["planning_seed"] == baseline_seed)
                  for name in protocol["checkpoint_conditions"]]
        lines.append(f"| {epoch} | " + " | ".join(f"{value}/5" for value in values) + " |")
    lines += ["", f"![Checkpoint 成功数]({output / 'checkpoint_success.png'})"]
    numerical = study["historical_cpu_checkpoint_comparison"]
    if numerical is not None:
        lines += ["", "### 与历史 CPU 训练 checkpoint 的数值差异", "",
                  f"比较运算设备：`{numerical.get('audit_device', '未记录')}`；历史文件：`{numerical.get('reference', '未记录')}`。",
                  "这是本次 GPU 训练结果与历史 CPU 训练结果的参数对照，不是一次新 CPU 训练，也不等同于同 checkpoint 的逐步推理对照。", "",
                  "| 矩阵 | 最大绝对差 | RMS 差 | 相对 Frobenius 差 |",
                  "| --- | ---: | ---: | ---: |"]
        for name, values in numerical.get("tensors", {}).items():
            lines.append(f"| {name} | {format_value(values.get('max_absolute'))} | {format_value(values.get('rms'))} | "
                         f"{format_value(values.get('relative_frobenius'))} |")
    lines += control_audit_markdown(study)
    lines += ["", "## 2. 最终 checkpoint：按规划种子报告", "",
              "### 条件设置", "",
              "| 条件 | G checkpoint | 方法 | 第一步门控 | 噪声 | snap | cache |",
              "| --- | --- | --- | --- | ---: | --- | --- |"]
    for group in final:
        condition = group["condition"]
        overrides = condition["overrides"]
        first = overrides.get("gcml_first_step_affordance_mode", "observed_graph")
        sigma = "—" if condition["method"] == "gcml_greedy" else format_value(overrides.get("gcml_noise_scale", .2))
        lines.append(f"| {LABELS.get(condition['name'], condition['name'])} | {condition['checkpoint_variant']} | {condition['method']} | "
                     f"{first} | {sigma} | {overrides.get('gcml_latent_projection', 'none')} | {overrides.get('gcml_cache_mode', 'none')} |")
    lines += ["", "### 各规划种子的五任务结果", "",
              "未记录的种子显示为“—”，不当作失败或成功；每一格只汇总该种子下的五个任务。", "",
              "| 条件 | " + " | ".join(f"seed {seed} 成功/5" for seed in seeds) + " | 平均 SPL | 平均环境动作数 |",
              "| --- | " + " | ".join("---:" for _ in seeds) + " | ---: | ---: |"]
    for group in final:
        blocks = {block["planning_seed"]: block for block in group["per_seed"]}
        cells = [f"{blocks[seed]['successes']}/5" if seed in blocks else "—" for seed in seeds]
        lines.append(f"| {LABELS.get(group['condition']['name'], group['condition']['name'])} | " + " | ".join(cells)
                     + f" | {group['mean_spl']:.4f} | {group['mean_steps']:.1f} |")
    lines += ["", "少量环境动作可能只来自起点连接与 STOP；失败时动作少不表示规划效率更高。greedy 没有候选轨迹，不能把零候选解释成采样失败。", "",
              "### 每个固定任务的成功频率", "",
              "表格为“成功次数／该条件实际运行的规划种子数”，用于查看失败是否集中于同一路径。", "",
              "| 条件 | " + " | ".join(f"任务 {task['task_index']:04d}／{task['episode_id']}" for task in final[0]["per_task"]) + " |",
              "| --- | " + " | ".join("---:" for _ in range(5)) + " |"]
    for group in final:
        lines.append(f"| {LABELS.get(group['condition']['name'], group['condition']['name'])} | "
                     + " | ".join(f"{task['successes']}/{task['planning_seed_repeats']}" for task in group["per_task"]) + " |")
    lines += ["", f"![机制比较]({output / 'mechanism_success.png'})", "", "## 3. 配对变化", "",
              "仅配对相同任务和规划种子；下面计数仍是重复执行，没有据此计算独立样本显著性。", "",
              "| 变化 | 匹配重复执行数 | 失败→成功 | 成功→失败 | 成功状态不变 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for pair in paired:
        lines.append(f"| {pair['label']} | {pair['matched_seed_task_repetitions']} | {pair['improved_repetitions']} | "
                     f"{pair['worsened_repetitions']} | {pair['unchanged_repetitions']} |")
    lines += ["", "## 4. 想象失败与独立图审计", "",
              "计数累加该条件全部规划种子。非法数指至少含一处非法转移的候选数；虚假到达指模型预测到达而独立图审计未到达。", "",
              "| 条件 | 执行次数 | task no_plan | 新搜索 no_plan/无候选 | 候选数 | 非法候选 | 虚假到达 | 回环 | 空门控 | 深度耗尽 | 状态未关联 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in final:
        counts, term = group["planning_counts"], group["candidate_terminations"]
        lines.append(f"| {LABELS.get(group['condition']['name'], group['condition']['name'])} | {group['seed_task_repetitions']} | "
                     f"{group['no_plan_task_repetitions']} | {group['fresh_no_plan_or_no_candidates_calls']} | {counts['candidates']} | "
                     f"{counts['invalid_candidates']} | {counts['false_predicted_arrival_candidates']} | {term.get('retrieved_state_cycle', 0)} | "
                     f"{term.get('empty_affordance', 0)} | {term.get('horizon_exhausted', 0)} | {term.get('state_unassociated', 0)} |")
    lines += ["", "| 条件 | 选中完整候选 | 选中虚假到达 | 非法执行选择 | 想象转移数 | 记录的 planner 秒数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in final:
        counts = group["planning_counts"]
        lines.append(f"| {LABELS.get(group['condition']['name'], group['condition']['name'])} | {counts['selected_candidates']} | "
                     f"{counts['selected_false_predicted_arrival']} | {counts['invalid_selected_edges']} | {counts['model_transitions']} | "
                     f"{group['observed_planner_call_seconds']:.3f} |")
    lines += ["", "选中候选计数仅包含新搜索；缓存后缀执行单独统计。planner 时间来自记录的调用计时，设备及并发负载可能不同，不能直接当作严格延迟基准。", "",
              "## 5. 旧计划后缀缓存", "",
              "当前协议先进行完整预算的新搜索，仅在新搜索无计划时使用旧后缀；因此 cache 没有免除该次重规划开销。", "",
              "| 条件 | 新搜索调用 | 存入计划 | 回退尝试 | 缓存未命中 | 回退选择边 | 回退完成边 | 非法回退 | 存入过计划的执行次数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in final:
        cache = group["cache"]
        if cache is None:
            continue
        counts = cache["counts"]
        lines.append(f"| {LABELS.get(group['condition']['name'], group['condition']['name'])} | {counts['fresh_planning_calls']} | "
                     f"{counts['plans_stored']} | {counts['fallback_attempts']} | {counts['fallback_cache_misses']} | "
                     f"{counts['fallback_actions_selected']} | {counts['fallback_actions_completed']} | {counts['invalid_fallback_actions']} | "
                     f"{cache['task_repetitions_with_plan_stored']}/{group['seed_task_repetitions']} |")
    for group in final:
        if group["cache"] is not None:
            cache = group["cache"]
            lines += ["", f"- **{LABELS.get(group['condition']['name'], group['condition']['name'])}**：失效原因 "
                      f"`{json.dumps(cache['invalidation_reasons'], ensure_ascii=False, sort_keys=True)}`。"
                      + (" 没有任何计划被存入，故本条件没有实际检验缓存收益的机会。" if cache["no_cache_opportunity"] else "")]
    suffix_audit = study["control_audits"].get("noise_suffix_opportunities")
    if suffix_audit is not None:
        data = suffix_audit["data"]
        low, moderate = data["summary_by_noise"]["0.02"], data["summary_by_noise"]["0.05"]
        actual = data["cache_005_completed_aggregate"]["counts"]
        lines += ["", f"旧日志审计确认：0.05 基线有 {moderate['matching_cached_suffix_opportunities']} 个匹配后缀机会，"
                  f"实际缓存运行的 {actual['fallback_attempts']} 次回退尝试均未选中后缀（选中 {actual['fallback_actions_selected']} 次）；"
                  f"0.02 基线则存在 {low['matching_cached_suffix_opportunities']} 个匹配机会，其中 {low['opportunities_with_graph_valid_goal_reaching_suffix']} 个后缀经离线图审计连续且通向目标。"
                  "这些后缀只证明当时有可复用路线，不保证缓存策略最终成功，离线图审计也没有参与未来动作筛除；"
                  f"证据见[旧计划后缀机会审计]({suffix_audit['json_path']})，其 JSON/Markdown 已计入输入 SHA256。"]
    controls = [group for group in groups if group["condition"].get("cohort") == "control"
                and group["condition"]["analysis_phase"] == "prespecified"]
    if controls:
        lines += ["", "### 附加控制任务", "", "| 条件 | Epoch | 规划种子与成功/5 | 平均 SPL |",
                  "| --- | ---: | --- | ---: |"]
        for group in controls:
            seed_text = "; ".join(f"{block['planning_seed']}: {block['successes']}/5" for block in group["per_seed"])
            lines.append(f"| {group['condition']['name']} | {group['condition']['epoch']} | {seed_text} | {group['mean_spl']:.4f} |")
    lines += exploratory_markdown(exploratory, study["run"])
    boundary_section = 7 if exploratory is not None else 6
    lines += ["", f"## {boundary_section}. 解释边界与可追溯产物", "",
              "- G 最小二乘使用经验图全部节点标签，是额外的全图监督诊断；成功不等于原在线 G 更新已经达到同样水平。汇点零标签表示已记录经验无出边，不证明物理环境无法离开。",
              "- 经验图辅助想象、真实当前出边、全 learned 第一动作分别标记。独立图审计不参与 learned 条件的未来动作筛除。",
              "- snap 只使用既定置信条件下的 Q 码本最近状态；cache 只保留此前模型生成的计划后缀。两者都应视作单独机制变化。",
              "- 新条件使用 CUDA 上预定的完整噪声表索引。相同种子不保证 CPU/CUDA 数值或随机序列逐位一致；本报告不宣称与历史 CPU 训练、推理 bit-exact。",
              "- 噪声为 0 时相同输入可能产生重复候选，64 个槽位不等于 64 个独立尝试。",
              "- 一个训练种子、一个场景、五个反复使用的开发任务，不足以给出跨场景泛化、独立测试集效果或训练随机性置信区间。",
              "- 没有按评测成绩自动早停；训练到预定最终 epoch。即使成功率不再变化，也不能单凭五任务证明所有读出都已收敛。", "",
              f"结构化结果：[comparison.json]({output / 'comparison.json'})；包含逐任务结果、每 seed 汇总、每 task 成功频率、所有 counter、配对变化和输入 SHA256。",
              f"冻结协议：[protocol.json]({Path(study['run']) / 'protocol.json'})。完整逐步日志保留在各 job 的原评估目录中，本脚本不生成新轨迹。", ""]
    return "\n".join(lines)

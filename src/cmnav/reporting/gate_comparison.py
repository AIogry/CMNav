"""Markdown presentation of computed gate comparison statistics."""
from __future__ import annotations
from typing import Any

def report_markdown(comparison: dict[str, Any]) -> str:
    text = ["# GCML 门控对照", "", f"状态：{comparison['status']}；checkpoint：`{comparison['checkpoint_label']}`。", "",
            "固定同一模型、同一经验图、同一 5 个任务和原先每 4 条边的教师子目标；K=64、H=96、noise=0.2，每任务总预算 500 个真实动作。",
            "教师已提供上层路线，此结果属于同场景特权诊断。首个想象动作始终使用当前节点真实出边 mask，并且不加 soft bonus。", "",
            "| 门控 | 阈值 | 成功 | 已完成阶段 | 非法候选 | 假到达候选 | 空门控 | 循环 | 局部解码源不匹配 | 未关联状态动作 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison["conditions"]:
        p = row["gcml_planning"]
        a = row["imagined_transition_source_audit"]
        t = p["candidate_terminations"]
        text.append(f"| {row['gate_mode']} | {row['threshold']:.2f} | {row['successes']}/{row['episodes']} | {row['stages']['completed']}/{row['stages']['total']} | {p['invalid_candidates']}/{p['candidates']} | {p['false_predicted_arrival_candidates']} | {t.get('empty_affordance',0)} | {t.get('retrieved_state_cycle',0)} | {a['decoded_source_mismatch']} | {a['unknown_off_manifold']} |")
    text += ["", "非法候选按从真实起点开始的经验图连续性统计。局部解码源检查逐步比较已关联想象状态与动作源；未关联状态另列为 unknown。首次非法动作之后即使局部源匹配，也不表示原物理路径恢复连续。", "",
             "各条件每次决策预算相同，决策次数和总想象动作量可能不同；完整分母见 comparison.json。未完成阶段不会计作成功。", "",
             f"0.50 原 checkpoint 复现检查：{comparison['baseline_replay_status']}。所有任务参数冻结校验通过：{comparison['all_parameters_unchanged']}。", ""]
    if comparison.get("paired_hard_replay_status") != "not_applicable":
        text += [f"新代码 hard 对旧代码同 checkpoint 的逐任务复现：{comparison['paired_hard_replay_status']}。", "",
                 "Soft 固定 center=0.5、temperature=0.1、weight=0.5；depth≥1 全部 495 条动作参与，使用归一化 utility + 0.5·logsigmoid((Gz−0.5)/0.1) + 原 Gaussian noise。",
                 "这是纯 soft 公式对照：真实汇点也会继续提议动作，未增加 abstain 或未来图过滤。因此空门控计数自然消失不能解释为质量改善。",
                 "Soft depth≥1 日志保存选中动作分数分解、全局最大值/rank 和总 logit 的 top8；未保存全部动作分数。Hard 日志保持原格式。", ""]
    return "\n".join(text)


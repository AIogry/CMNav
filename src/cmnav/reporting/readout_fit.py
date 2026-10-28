"""Render the frozen-G fitting report from already computed statistics."""
from __future__ import annotations
from pathlib import Path


def write_readout_report(directory: Path, source_dir: Path, ls_summaries: list,
                         condition_summaries: list, *, epochs: int, source_batch_count: int,
                         source_exposure_count: int, sink_count: int) -> None:
    lines = ["# 固定 Q 后的 G 读出拟合", "", f"源 checkpoint：`{source_dir}`。",
             "Q/V/W 全程冻结；没有导航、没有新增环境经验。原始联合训练计数保留，G-only 训练单独记录。",
             "", "## 线性表达能力参考", "",
             "| 参考 | rank | condition | float32 源 RMSE | 源 FP/FN | 汇点 FP | G Frobenius |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for reference in ls_summaries:
        s = reference["solver"]
        r = reference["float32_metrics"]
        lines.append(f"| {reference['variant']} | {s['rank']} | {s['condition_largest_over_smallest']:.6g} | {r['sources']['rmse']:.6g} | {r['sources']['fp']}/{r['sources']['fn']} | {r['sinks']['fp']} | {reference['G_float32_norms']['frobenius']:.6g} |")
    lines += ["", "最小范数解会删除未受监督方向上的原始随机分量；minimum-change 解保留这些分量，更接近从原 G 出发的无正则固定特征梯度下降极限。两者的汇点表现不能混为一谈。",
              "", f"## 同源预算的固定 {epochs} 轮迭代", "",
              "| 条件 | 源 RMSE | 源正标签 RMSE | 源 FP/FN | 汇点 RMSE | 汇点 FP |",
              "|---|---:|---:|---:|---:|---:|"]
    for condition in condition_summaries:
        s, t = condition["final_metrics"]["sources"], condition["final_metrics"]["sinks"]
        lines.append(f"| {condition['variant']} | {s['rmse']:.6g} | {s['positive_rmse']:.6g} | {s['fp']}/{s['fn']} | {t['rmse']:.6g} | {t['fp']} |")
    lines += ["", f"每个条件重放相同 {source_batch_count} 个原始片段、每轮 {source_exposure_count} 个源样本，共 {epochs} 轮。所有条件使用同一批次顺序，从原 checkpoint 的同一 G 开始。",
              f"补汇点条件每轮额外增加一个 {sink_count} 状态的全零标签批次，总计额外 {epochs * sink_count} 次状态曝光、{epochs} 次更新；它与源状态条件匹配源预算，不匹配总预算。",
              "", "## 解释边界", "",
              "这些是同一已知经验图上的线性读出与优化诊断，不是新场景泛化或导航成功结果。",
              "来源状态集合来自实际训练转移，汇点的零标签仅表示没有已观察出边。",
              "逐轮实际学习率、截断计数、正负误差与分类保存在 readout_epochs.jsonl 和每个条件的 readout_training.json。",
              "SVD 奇异谱、rcond、最小范数选择与 float64/float32 差异均保存在 least_squares/ 中；算子范数仅提供输入漂移放大上界，没有伪称测量了新的想象轨迹。", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")

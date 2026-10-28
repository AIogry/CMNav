"""Presentation labels for saved navigation and mechanism comparisons."""
from __future__ import annotations



NAVIGATION_LABELS = {"cml": "CML", "dijkstra": "Dijkstra", "oracle_follower": "Full oracle"}
NAVIGATION_COLORS = {"cml": "#D55E00", "dijkstra": "#0072B2", "oracle_follower": "#009E73"}
ACTION_COLORS = ("#56B4E9", "#CC79A7", "#E69F00", "#444444")
GCML_LABELS = {"gcml": "GCML learned G", "gcml_graph_affordance": "GCML graph affordance",
          "gcml_greedy": "GCML greedy", "dijkstra": "Dijkstra"}
GCML_SHORT_LABELS = {"gcml": "学得 G", "gcml_graph_affordance": "经验图辅助", "gcml_greedy": "Greedy", "dijkstra": "Dijkstra"}
GCML_COLORS = {"gcml": "#c73e33", "gcml_graph_affordance": "#159b80", "gcml_greedy": "#9557aa", "dijkstra": "#2675b7"}
MODE_LABELS = {"gcml": "学得 G", "gcml_graph_affordance": "经验图可用性辅助"}
FIGURE_LABELS = {"gcml": "Learned G", "gcml_graph_affordance": "Observed-graph\naffordance"}
MECHANISM_LABELS = {
    "raw_imagination": "原 G／想象",
    "fit_imagination": "充分拟合 G／想象",
    "graph_greedy": "当前经验图／greedy",
    "raw_learned_greedy": "原 G 当前门控／greedy",
    "fit_learned_greedy": "拟合 G 当前门控／greedy",
    "raw_pure_imagination": "原 G／第一步也学得门控",
    "fit_pure_imagination": "拟合 G／第一步也学得门控",
    "fit_graph_imagination": "经验图辅助／想象",
    "minimum_norm_imagination": "最小范数 G／想象",
    "fit_noise_0.1": "拟合 G／噪声 0.1",
    "fit_noise_0.05": "拟合 G／噪声 0.05",
    "fit_noise_0.02": "拟合 G／噪声 0.02",
    "fit_noise_0": "拟合 G／噪声 0",
    "fit_snap_0.05": "拟合 G／0.05＋snap",
    "fit_cache_0.05": "拟合 G／0.05＋cache",
    "fit_snap_cache_0.05": "拟合 G／0.05＋snap＋cache",
}

"""Offline graph/model contracts and bounded task execution without Habitat."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cmnav.evaluation.oracle_graph_navigation import evaluate_graph_navigation_task
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.methods.cml import CognitiveMapLearner


def distance(source, target):
    return float(np.linalg.norm(np.asarray(source) - target))


def test_graph_rejects_geodesically_separated_neighbours():
    graph = OracleObservedGraph(merge_radius=0.2)
    first = graph.add_position([0.0, 0.0, 0.0], distance)
    second = graph.add_position([0.1, 0.0, 0.0], lambda _source, _target: float("inf"))
    assert (first, second) == (0, 1)
    assert graph.add_position([0.01, 0.0, 0.0], distance) == first
    queries = []
    assert graph.nearest([5.0, 0.0, 0.0], lambda a, b: queries.append((a, b)), radius=0.2) is None
    assert queries == []


def test_graph_roundtrip_preserves_directed_occurrences_and_mean_cost():
    graph = OracleObservedGraph(merge_radius=0.2)
    for point in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]):
        graph.add_position(point, distance)
    assert graph.add_transition(0, 1, trajectory_id="a", step=0, cost=1.0) == 0
    assert graph.add_transition(0, 1, trajectory_id="b", step=3, cost=3.0) == 0
    assert graph.add_transition(1, 2, trajectory_id="b", step=4, cost=0.5) == 1
    assert graph.add_transition(2, 2, trajectory_id="b", step=5, cost=0.0) is None
    restored = OracleObservedGraph.from_state_dict(json.loads(json.dumps(graph.state_dict())))
    assert restored.state_dict() == graph.state_dict()
    assert restored.observed_edge_indices == [0, 0, 1]
    assert restored.edge_counts == [2, 1]
    assert restored.edge_mean_costs == [2.0, 0.5]
    assert restored.shortest_path(0, 2) == [0, 1]
    assert restored.shortest_path(2, 0) is None
    invalid = restored.state_dict()
    invalid["edge_counts"][0] += 1
    with pytest.raises(ValueError, match="counts"):
        OracleObservedGraph.from_state_dict(invalid)


def test_cml_roundtrip_preserves_seeded_updates_and_choices():
    edges = [(0, 1), (1, 2), (1, 3), (2, 0)]
    model = CognitiveMapLearner(4, edges, dimension=8, seed=7)
    sequence = [(0, 0, 1), (1, 1, 2), (2, 3, 0), (0, 0, 1), (1, 2, 3)]
    model.fit([sequence], epochs=3)
    restored = CognitiveMapLearner.from_state_dict(json.loads(json.dumps(model.state_dict())))
    assert restored.state_dict() == model.state_dict()
    for state in range(4):
        candidates = [index for index, edge in enumerate(edges) if edge[0] == state]
        for goal in range(4):
            assert torch.equal(restored.action_scores(state, goal), model.action_scores(state, goal))
            for used in ([], candidates):
                assert restored.choose_edge(state, goal, candidates, used) == model.choose_edge(state, goal, candidates, used)
    assert restored.fit([sequence], epochs=1) == model.fit([sequence], epochs=1)
    assert restored.state_dict() == model.state_dict()


class FakeOracleEnvironment:
    """Deterministic linear task with real fake transitions, not a simulator."""

    def __init__(self, budget):
        self.budget = budget
        self.goal = [0.5, 0.0, 0.0]
        self.steps = 0
        self.stopped = False
        self.position = [0.0, 0.0, 0.0]

    def observation(self):
        return {"rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "depth": np.ones((4, 4), dtype=np.float32),
                "pointgoal": np.asarray(self.goal, dtype=np.float32)}

    def reset(self):
        self.steps, self.stopped = 0, False
        self.position = [0.0, 0.0, 0.0]
        return self.observation()

    def episode_info(self):
        return {"episode_id": "fake-episode", "scene_id": "fake-scene"}

    def privileged_state(self):
        return {"position": list(self.position), "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "goal_position": list(self.goal)}

    @property
    def reference_target(self):
        return list(self.goal)

    @property
    def episode_over(self):
        return self.stopped or self.steps >= self.budget

    def geodesic_distance(self, source, target):
        return distance(source, target)

    def get_reference_action(self, target_position=None):
        target = self.goal if target_position is None else target_position
        return "STOP" if distance(self.position, target) < 0.01 else "MOVE_FORWARD"

    def step(self, action):
        assert not self.episode_over
        if action == "MOVE_FORWARD":
            self.position[0] += 0.25
        elif action == "STOP":
            self.stopped = True
        else:
            raise AssertionError(f"Unexpected fake action {action}")
        self.steps += 1
        return self.observation()

    def metrics(self):
        success = float(self.stopped and distance(self.position, self.goal) < 0.2)
        return {"success": success, "spl": success, "distance_to_goal": distance(self.position, self.goal),
                "collisions": {"count": 0, "is_collision": False}, "num_steps": self.steps}


@pytest.mark.parametrize("budget, expected_success, expected_reason, expected_steps", [
    (5, 1.0, "goal_stop", 3),
    (2, 0.0, "episode_budget", 2),
])
def test_evaluator_preserves_stop_and_global_budget(
    tmp_path, budget, expected_success, expected_reason, expected_steps,
):
    graph = OracleObservedGraph(merge_radius=0.1)
    for point in ([0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]):
        graph.add_position(point, distance)
    graph.add_transition(0, 1, trajectory_id="train", step=0, cost=0.25)
    graph.add_transition(1, 2, trajectory_id="train", step=1, cost=0.25)
    before = graph.state_dict()
    config = {"base": {"environment": {"rgbd": {"max_depth": 10.0}}},
              "attachment_radius": 0.1, "local_goal_radius": 0.01,
              "max_local_steps": 8, "max_graph_hops": 10}
    result = evaluate_graph_navigation_task(config, FakeOracleEnvironment(budget), graph, None,
                                            "dijkstra", tmp_path / "task", 7)
    assert result["metrics"]["success"] == expected_success
    assert result["end_reason"] == expected_reason
    assert result["steps"] == expected_steps
    assert graph.state_dict() == before
    lines = [json.loads(line) for line in (tmp_path / "task" / "execution.jsonl").read_text().splitlines()]
    assert len(lines) == expected_steps + 1
    assert sum(line["action"] == "STOP" for line in lines) == int(expected_success)

"""Directed experience graph with explicitly privileged state association.

Nodes are positions actually supplied by a rollout. NavMesh distance is used
only to guard association of nearby observed positions; it never creates an
edge. Edges are supplied observed transitions, with every occurrence retained.
This is an oracle-state diagnostic representation, not visual localization.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cmnav.data.trajectories import TrajectoryReader
from cmnav.memory.graph_search import shortest_path
from cmnav.runtime.artifacts import sha256_file, write_json


GeodesicDistance = Callable[[list[float], list[float]], float]


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite real number.") from exc
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        interval = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {interval}.")
    return result


def _position(value: Any) -> list[float]:
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError("Position must contain three finite real coordinates.") from exc
    if len(values) != 3 or any(
        isinstance(item, bool) or not isinstance(item, Real) for item in values
    ):
        raise ValueError("Position must contain three finite real coordinates.")
    try:
        result = [float(item) for item in values]
    except (ValueError, OverflowError) as exc:
        raise ValueError("Position must contain three finite real coordinates.") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Position must contain three finite real coordinates.")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return int(value)


class OracleObservedGraph:
    """Observed directed transitions with pose-assisted node association.

    Position association checks Euclidean candidates in nearest-first order,
    then requires the provided geodesic distance to be within the same radius.
    Node representatives remain the first actual observation, without averaging
    positions into potentially unobserved geometry. Node IDs do not encode
    orientation; a privileged local controller must handle heading differences.

    Repeated transitions share one edge ID but keep separate provenance and
    cost records. Shortest paths minimize each edge's mean observed cost; the
    caller must use one consistent, nonnegative cost unit throughout a graph.
    Public list properties return detached copies to protect its invariants.
    """

    SCHEMA_VERSION = 1

    def __init__(self, merge_radius: float = 0.2) -> None:
        self.merge_radius = _finite_number(merge_radius, "merge_radius", positive=True)
        self._positions: list[list[float]] = []
        self._edges: list[tuple[int, int]] = []
        self._edge_lookup: dict[tuple[int, int], int] = {}
        self._outgoing: list[list[int]] = []
        self._observed_edge_indices: list[int] = []
        self._occurrences: list[dict[str, Any]] = []
        self._edge_counts: list[int] = []
        self._edge_mean_costs: list[float] = []

    @property
    def positions(self) -> list[list[float]]:
        return [list(position) for position in self._positions]

    @property
    def edges(self) -> list[tuple[int, int]]:
        return list(self._edges)

    @property
    def observed_edge_indices(self) -> list[int]:
        return list(self._observed_edge_indices)

    @property
    def occurrences(self) -> list[dict[str, Any]]:
        return deepcopy(self._occurrences)

    @property
    def edge_counts(self) -> list[int]:
        return list(self._edge_counts)

    @property
    def edge_mean_costs(self) -> list[float]:
        return list(self._edge_mean_costs)

    def add_position(self, position: Any, geodesic: GeodesicDistance) -> int:
        """Associate with a nearby reachable node, or add this observed point."""
        observed = _position(position)
        existing = self.nearest(observed, geodesic, self.merge_radius)
        if existing is not None:
            return existing
        index = len(self._positions)
        self._positions.append(observed)
        self._outgoing.append([])
        return index

    def nearest(
        self,
        position: Any,
        geodesic: GeodesicDistance,
        radius: float,
    ) -> int | None:
        """Return the nearest eligible observed node, with bounded oracle queries.

        Only points within Euclidean ``radius`` invoke the geodesic callback.
        Positive infinity means unreachable and is skipped. NaN, negative, or
        nonnumeric distance results indicate a broken distance provider.
        """
        query = _position(position)
        limit = _finite_number(radius, "radius")
        if not callable(geodesic):
            raise ValueError("geodesic must be callable.")
        candidates = sorted(
            (distance, index)
            for index, candidate in enumerate(self._positions)
            if (distance := math.dist(query, candidate)) <= limit
        )
        for _, index in candidates:
            result = geodesic(list(query), list(self._positions[index]))
            if isinstance(result, bool) or not isinstance(result, Real):
                raise ValueError("Geodesic callback must return a nonnegative real distance.")
            try:
                distance = float(result)
            except (OverflowError, ValueError) as exc:
                raise ValueError("Geodesic callback returned an invalid distance.") from exc
            if math.isnan(distance) or distance < 0.0:
                raise ValueError("Geodesic callback returned an invalid distance.")
            if math.isfinite(distance) and distance <= limit:
                return index
        return None

    def add_transition(
        self,
        source: int,
        target: int,
        *,
        trajectory_id: str,
        step: int,
        cost: float,
    ) -> int | None:
        """Record one actual directed transition; collapsed self edges are skipped.

        The caller is responsible for passing consecutive states from one
        continuous trajectory. No reverse edges or reset transitions are added.
        Step identifies the source transition within that trajectory.
        """
        source = self._node(source)
        target = self._node(target)
        if not isinstance(trajectory_id, str) or not trajectory_id.strip():
            raise ValueError("trajectory_id must be a nonempty string.")
        step = _nonnegative_integer(step, "step")
        cost = _finite_number(cost, "cost")
        if source == target:
            return None
        pair = (source, target)
        edge = self._edge_lookup.get(pair)
        if edge is None:
            edge = len(self._edges)
            self._edges.append(pair)
            self._edge_lookup[pair] = edge
            self._outgoing[source].append(edge)
            self._edge_counts.append(0)
            self._edge_mean_costs.append(0.0)
        count = self._edge_counts[edge] + 1
        previous_mean = self._edge_mean_costs[edge]
        self._edge_counts[edge] = count
        self._edge_mean_costs[edge] = previous_mean + (cost - previous_mean) / count
        self._observed_edge_indices.append(edge)
        self._occurrences.append(
            {
                "edge_index": edge,
                "trajectory_id": trajectory_id,
                "step": step,
                "cost": cost,
            }
        )
        return edge

    def outgoing(self, node: int) -> list[int]:
        return list(self._outgoing[self._node(node)])

    def edge_target(self, edge: int) -> int:
        return self._edges[self._edge(edge)][1]

    def shortest_path(self, start: int, goal: int) -> list[int] | None:
        """Dijkstra path over observed directed edges, or None when disconnected."""
        start = self._node(start)
        goal = self._node(goal)
        path, _ = shortest_path(len(self._positions), self._edges,
                                self._edge_mean_costs, start, goal)
        return path

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-compatible geometry, occurrences, and checked aggregates."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "kind": "oracle_observed_graph",
            "merge_radius": self.merge_radius,
            "positions": self.positions,
            "edges": [list(edge) for edge in self._edges],
            "observed_edge_indices": self.observed_edge_indices,
            "occurrences": self.occurrences,
            "edge_counts": self.edge_counts,
            "edge_mean_costs": self.edge_mean_costs,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> OracleObservedGraph:
        """Validate and reconstruct all edges and aggregates from provenance.

        Deserialization does not invoke a simulator or verify physical truth;
        the source dataset must provide that provenance. Redundant aggregate
        fields must agree with the recorded occurrences.
        """
        if not isinstance(state, dict):
            raise ValueError("Graph state must be a dictionary.")
        required = {
            "schema_version", "kind", "merge_radius", "positions", "edges",
            "observed_edge_indices", "occurrences", "edge_counts", "edge_mean_costs",
        }
        if not required.issubset(state):
            raise ValueError(f"Graph state is missing fields: {sorted(required - state.keys())}.")
        version = _nonnegative_integer(state["schema_version"], "schema_version")
        if version != cls.SCHEMA_VERSION or state["kind"] != "oracle_observed_graph":
            raise ValueError("Unsupported observed graph kind or schema version.")
        for key in ("positions", "edges", "observed_edge_indices", "occurrences", "edge_counts", "edge_mean_costs"):
            if not isinstance(state[key], list):
                raise ValueError(f"Graph state {key} must be a list.")
        graph = cls(state["merge_radius"])
        graph._positions = [_position(position) for position in state["positions"]]
        graph._outgoing = [[] for _ in graph._positions]
        edges = []
        for pair in state["edges"]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each serialized edge must contain source and target indices.")
            source, target = graph._node(pair[0]), graph._node(pair[1])
            if source == target:
                raise ValueError("Serialized observed graph must not contain self edges.")
            edges.append((source, target))
        if len(set(edges)) != len(edges):
            raise ValueError("Serialized graph edges must be unique directed pairs.")
        indices = [
            _nonnegative_integer(value, "observed edge index")
            for value in state["observed_edge_indices"]
        ]
        if len(indices) != len(state["occurrences"]):
            raise ValueError("Observed edge indices and occurrence counts disagree.")
        for expected, occurrence in zip(indices, state["occurrences"]):
            if not isinstance(occurrence, dict) or not {
                "edge_index", "trajectory_id", "step", "cost"
            }.issubset(occurrence):
                raise ValueError("Malformed observed-edge provenance record.")
            edge = _nonnegative_integer(occurrence["edge_index"], "occurrence edge index")
            if edge != expected or edge >= len(edges):
                raise ValueError("Occurrence edge index is out of range or inconsistent.")
            source, target = edges[edge]
            actual = graph.add_transition(
                source,
                target,
                trajectory_id=occurrence["trajectory_id"],
                step=occurrence["step"],
                cost=occurrence["cost"],
            )
            if actual != edge:
                raise ValueError("Edge order does not match the order of first observation.")
        if graph._edges != edges:
            raise ValueError("Every serialized edge must have recorded observations.")
        counts = [_nonnegative_integer(value, "edge count") for value in state["edge_counts"]]
        if counts != graph._edge_counts:
            raise ValueError("Serialized edge counts disagree with provenance.")
        means = [_finite_number(value, "edge mean cost") for value in state["edge_mean_costs"]]
        if len(means) != len(graph._edge_mean_costs) or any(
            not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in zip(graph._edge_mean_costs, means)
        ):
            raise ValueError("Serialized mean edge costs disagree with provenance.")
        return graph

    def _node(self, value: Any) -> int:
        index = _nonnegative_integer(value, "node index")
        if index >= len(self._positions):
            raise ValueError(f"Node index {index} is outside the observed graph.")
        return index

    def _edge(self, value: Any) -> int:
        index = _nonnegative_integer(value, "edge index")
        if index >= len(self._edges):
            raise ValueError(f"Edge index {index} is outside the observed graph.")
        return index


def build_observed_graph(config: dict[str, Any], directory: Path,
                 summaries: list[dict[str, Any]], geodesic: GeodesicDistance,
                 *, plan_id: str, reference_directory: Path | None = None) -> OracleObservedGraph:
    """Build from recorded learning transitions with an explicit oracle callback.

    ``geodesic`` must refer to the recorded scene and remain usable for the
    duration of this call. This builder does not create, reset, or close an
    environment. The caller supplies the declared privileged association
    service; no reset boundary or evaluation transition creates an edge.
    """
    graph = OracleObservedGraph(merge_radius=config["merge_radius"])
    source = reference_directory or directory
    associations = []
    for summary in summaries:
        if summary["status"] != "complete":
            continue
        name = summary["trajectory_id"]
        reader = TrajectoryReader(source / "reference" / name)
        expected = {
            "trajectory_id": name, "scene_id": summary["scene_id"],
            "source_episode_id": summary["episode_id"], "plan_id": plan_id,
            "collection_protocol": "goal_directed_official_follower_v1",
            "policy": "privileged_habitat_shortest_path_follower",
            "sensor_spec": config["base"]["environment"]["rgbd"],
            "action_spec": {"forward_step_size": config["base"]["environment"]["forward_step_size"],
                            "turn_angle": config["base"]["environment"]["turn_angle"]},
        }
        if any(reader.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Reference trajectory metadata differs from its task/configuration")
        with (source / "audit" / f"{name}.jsonl").open(encoding="utf-8") as stream:
            states = [json.loads(line) for line in stream]
        if len(states) != reader.summary["observation_count"]:
            raise RuntimeError("Privileged audit and RGB-D trajectory are not aligned")
        node_ids = []
        previous = None
        distance_since_node_change = 0.0
        # Reading transitions loads the actual saved RGB-D; state learning in
        # this diagnostic deliberately consumes the separate privileged audit.
        reader.observation(0)
        for offset, state in enumerate(states):
            if state["index"] != offset:
                raise RuntimeError("Nonconsecutive privileged audit index")
            node = graph.add_position(state["position"], geodesic)
            node_ids.append(node)
            if previous is not None:
                # Include motion absorbed by node merging. This is an observed
                # travel-cost surrogate, not a bound on executed controller cost.
                cost = float(np.linalg.norm(np.asarray(states[offset - 1]["position"])
                                            - np.asarray(state["position"])))
                distance_since_node_change += cost
                if node != previous:
                    graph.add_transition(previous, node, trajectory_id=name,
                                         step=offset - 1, cost=max(distance_since_node_change, 1e-8))
                    distance_since_node_change = 0.0
            previous = node
        for offset, transition in enumerate(reader.transitions()):
            if transition["action"] != states[offset + 1]["action"]:
                raise RuntimeError("Audit action differs from recorded trajectory")
        associations.append({"trajectory_id": name, "node_ids": node_ids,
                             "audit_sha256": sha256_file(source / "audit" / f"{name}.jsonl"),
                             "trajectory_manifest_sha256": sha256_file(reader.directory / "manifest.json")})
    if not graph.edges:
        raise RuntimeError("No observed translating edges; inspect reference failures")
    write_json(directory / "graph_associations.json", associations)
    write_json(directory / "graph.json", graph.state_dict())
    return graph

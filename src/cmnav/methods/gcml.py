"""PyTorch GCML learning and continuous latent imagination for PointNav.

This is a declared adaptation of LH-cbicr/GCML's abstract-graph notebook:
residual W learning, current-state aligned G supervision, bounded residual
steps, hard availability masks and complete-route cost selection. It does
not call a simulator, shortest-path search, or replace ``z + V[a]`` by an
edge's recorded destination. Observed-graph availability is an explicit
diagnostic mode; the main mode learns availability with G.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader


_SCHEMA = "cmnav.gcml.v1"
_ALGORITHM = "gcml_residual_readouts_continuous_imagination_v1"
_STEP_RULE = "eta_min_requested_and_safety_times_batch_over_frobenius_squared"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return result


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"nonfinite GCML {name}")


def _generator_hash(generator: torch.Generator) -> str:
    return hashlib.sha256(bytes(generator.get_state().tolist())).hexdigest()


class GenerativeCognitiveMapLearner:
    """Q/V local transition memory, residual W/G readouts, and rollouts.

    A trajectory is one batch. All residuals use its pre-update Q values;
    batches must contain unique target/action IDs. The residual step cap is
    explicit: eta_eff <= safety / (||X||_F² / B). Since this upper-bounds
    lambda_max(XXᵀ/B), safety < 2 prevents divergence of a frozen-batch
    least-squares step. This is not a convergence proof for joint Q/V/W/G.

    Gaussian initial scales Q=1, V=W=G=.1 and CPU private RNGs are used.
    Every fit consumes all supplied trajectories, with a fresh permutation
    each epoch. Imagination never writes the learned tensors.
    """

    def __init__(
        self, n_states: int, edges: Sequence[tuple[int, int]],
        dimension: int = 1000, seed: int = 0, learning_rate: float = .01,
        *, state_learning_rate: float = .1, inverse_learning_rate: float = .001,
        affordance_learning_rate: float = .0005, readout_step_safety: float = 1.8,
        reference_metadata: Optional[Mapping[str, Any]] = None,
        device: str | torch.device = "cpu",
    ):
        self.n_states = _integer(n_states, "n_states", 1)
        self.dimension = _integer(dimension, "dimension", 1)
        self.seed = _integer(seed, "seed")
        if self.seed > 2**64 - 1:
            raise ValueError("seed must fit an unsigned 64-bit integer")
        self.learning_rate = _number(learning_rate, "learning_rate")
        self.state_learning_rate = _number(state_learning_rate, "state_learning_rate")
        self.inverse_learning_rate = _number(inverse_learning_rate, "inverse_learning_rate")
        self.affordance_learning_rate = _number(affordance_learning_rate, "affordance_learning_rate")
        self.readout_step_safety = _number(readout_step_safety, "readout_step_safety")
        if self.readout_step_safety >= 2:
            raise ValueError("readout_step_safety must be strictly less than 2")
        self.reference_metadata = dict(reference_metadata or {})
        parsed = []
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("edges must contain (source, target) pairs")
            parsed.append((self._state(edge[0]), self._state(edge[1])))
        self.edges = tuple(parsed)
        if not self.edges or len(set(self.edges)) != len(self.edges):
            raise ValueError("GCML requires nonempty unique directed edges")
        self.n_edges = len(self.edges)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("GCML device must be cpu or cuda")
        initial = torch.Generator(device="cpu").manual_seed(self.seed)
        def normal(shape: tuple[int, int], scale: float) -> torch.Tensor:
            return (scale * torch.randn(shape, generator=initial, dtype=torch.float32)).to(self.device)
        self.Q = normal((self.dimension, self.n_states), 1.)
        self.V = normal((self.dimension, self.n_edges), .1)
        self.W = normal((self.n_edges, self.dimension), .1)
        self.G = normal((self.n_edges, self.dimension), .1)
        self.device = self.Q.device
        self._shuffle_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._planning_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._availability = torch.zeros((self.n_edges, self.n_states), device=self.device)
        for action, (source, _) in enumerate(self.edges):
            self._availability[action, source] = 1.
        self._sources = torch.tensor([edge[0] for edge in self.edges], device=self.device)
        self._targets = torch.tensor([edge[1] for edge in self.edges], device=self.device)
        self._outgoing = tuple(tuple(i for i, edge in enumerate(self.edges) if edge[0] == s)
                               for s in range(self.n_states))
        self.updates = self.trajectory_updates = self.epochs_completed = self.planning_calls = 0

    def _state(self, value: Any) -> int:
        result = _integer(value, "state")
        if result >= self.n_states:
            raise ValueError("state is outside the GCML vocabulary")
        return result

    def _edge(self, value: Any) -> int:
        result = _integer(value, "action")
        if result >= self.n_edges:
            raise ValueError("action is outside the GCML vocabulary")
        return result

    def _legal(self, state: int, edges: Iterable[int]) -> list[int]:
        result = sorted({self._edge(edge) for edge in edges})
        if any(self.edges[action][0] != state for action in result):
            raise ValueError("provided current outgoing action has a different source")
        return result

    def reset_planning_rng(self, seed: int) -> None:
        """Start a task's reproducible RNG stream; do not call between steps."""
        seed = _integer(seed, "planning seed")
        if seed > 2**64 - 1:
            raise ValueError("planning seed must fit an unsigned 64-bit integer")
        self._planning_generator.manual_seed(seed)
        self.planning_calls = 0

    def _trajectories(self, trajectories: Iterable[Sequence[tuple[int, int, int]]]) -> list[torch.Tensor]:
        records = []
        for trajectory in trajectories:
            triples = []
            previous = None
            for transition in trajectory:
                if len(transition) != 3:
                    raise ValueError("each transition must contain source, action, target")
                source, action, target = self._state(transition[0]), self._edge(transition[1]), self._state(transition[2])
                if self.edges[action] != (source, target):
                    raise ValueError("transition disagrees with its directed edge")
                if previous is not None and source != previous:
                    raise ValueError("trajectory must be continuous; separate reset boundaries")
                triples.append((source, action, target))
                previous = target
            if not triples:
                raise ValueError("GCML requires nonempty training trajectories")
            if len({row[1] for row in triples}) != len(triples) or len({row[2] for row in triples}) != len(triples):
                raise ValueError("GCML trajectory batches require unique action and target IDs")
            records.append(torch.tensor(triples, dtype=torch.long))
        if not records:
            raise ValueError("GCML requires at least one training trajectory")
        return records

    def _effective_step(self, rate: float, features: torch.Tensor) -> float:
        energy = float(features.square().sum().item())
        return min(rate, self.readout_step_safety * features.shape[1] / max(energy, 1e-30))

    @torch.no_grad()
    def fit(self, trajectories: Iterable[Sequence[tuple[int, int, int]]], epochs: int = 100,
            *, progress_callback: Optional[Callable[[dict[str, Any]], None]] = None) -> dict[str, Any]:
        """Consume all balanced trajectory batches; return JSON-safe diagnostics."""
        epochs = _integer(epochs, "epochs", 1)
        if not self.reference_metadata.get("commit") or not self.reference_metadata.get("notebook_sha256"):
            raise ValueError("GCML training requires pinned reference_metadata commit and notebook_sha256")
        records = self._trajectories(trajectories)
        loader = DataLoader(records, batch_size=1, shuffle=True, num_workers=0, generator=self._shuffle_generator)
        initial_updates, initial_batches = self.updates, self.trajectory_updates
        epoch_records = []
        started = time.perf_counter()
        for _ in range(epochs):
            totals = {name: 0. for name in ("prediction_mse", "inverse_mse", "affordance_mse")}
            update_norms = {name: 0. for name in ("Q", "V", "W", "G")}
            rate_min = {"W": self.inverse_learning_rate, "G": self.affordance_learning_rate}
            rate_max = {"W": 0., "G": 0.}
            capped = {"W": 0, "G": 0}
            transition_weighted = {name: 0. for name in totals}
            for count, outer in enumerate(loader, 1):
                batch = outer[0].to(self.device)
                sources, actions, targets = batch[:, 0], batch[:, 1], batch[:, 2]
                # Advanced-indexed tensors are independent pre-update copies.
                current = self.Q[:, sources]
                difference = self.Q[:, targets] - current
                prediction_error = difference - self.V[:, actions]
                desired_actions = torch.nn.functional.one_hot(actions, num_classes=self.n_edges).T.to(self.Q.dtype)
                inverse_error = desired_actions - self.W @ difference
                affordance_error = self._availability[:, sources] - self.G @ current
                for name, value in (("prediction residual", prediction_error), ("inverse residual", inverse_error),
                                    ("affordance residual", affordance_error)):
                    _finite(value, name)
                w_rate = self._effective_step(self.inverse_learning_rate, difference)
                g_rate = self._effective_step(self.affordance_learning_rate, current)
                q_update = -self.state_learning_rate * prediction_error
                v_update = self.learning_rate * prediction_error
                w_update = (w_rate / len(batch)) * (inverse_error @ difference.T)
                g_update = (g_rate / len(batch)) * (affordance_error @ current.T)
                self.Q[:, targets] += q_update
                self.V[:, actions] += v_update
                self.W += w_update
                self.G += g_update
                for name, update in (("Q", q_update), ("V", v_update), ("W", w_update), ("G", g_update)):
                    _finite(getattr(self, name), f"{name} after update")
                    update_norms[name] += float(torch.linalg.vector_norm(update).item())
                for name, residual in (("prediction_mse", prediction_error), ("inverse_mse", inverse_error),
                                       ("affordance_mse", affordance_error)):
                    mse = float(residual.square().mean().item())
                    totals[name] += mse
                    transition_weighted[name] += mse * len(batch)
                for name, effective, requested in (("W", w_rate, self.inverse_learning_rate),
                                                   ("G", g_rate, self.affordance_learning_rate)):
                    rate_min[name] = min(rate_min[name], effective)
                    rate_max[name] = max(rate_max[name], effective)
                    capped[name] += int(effective < requested)
                self.updates += len(batch)
                self.trajectory_updates += 1
            self.epochs_completed += 1
            record = {
                "epoch": self.epochs_completed,
                **{name: value / len(records) for name, value in totals.items()},
                "transition_weighted": {name: value / sum(map(len, records)) for name, value in transition_weighted.items()},
                "parameter_frobenius_norms": {name: float(torch.linalg.vector_norm(getattr(self, name)).item()) for name in update_norms},
                "mean_batch_update_norms": {name: value / len(records) for name, value in update_norms.items()},
                "effective_readout_rate_min": rate_min, "effective_readout_rate_max": rate_max,
                "capped_readout_batches": capped, "finite": True,
                "elapsed_seconds": time.perf_counter() - started,
            }
            epoch_records.append(record)
            if progress_callback is not None:
                progress_callback(dict(record))
        return {
            "algorithm": _ALGORITHM, "backend": "torch", "dtype": "float32", "device": str(self.device),
            "reference": self.reference_metadata, "epochs": epochs, "epochs_completed": self.epochs_completed,
            "trajectories_per_epoch": len(records), "transitions_per_epoch": sum(map(len, records)),
            "updates_added": self.updates - initial_updates, "updates_total": self.updates,
            "update_count_unit": "consumed_transitions", "trajectory_updates_added": self.trajectory_updates - initial_batches,
            "trajectory_updates_total": self.trajectory_updates, "repeated_index_policy": "require_unique_action_and_target_per_batch",
            "readout_step_rule": _STEP_RULE, "readout_step_safety": self.readout_step_safety,
            "requested_inverse_learning_rate": self.inverse_learning_rate,
            "requested_affordance_learning_rate": self.affordance_learning_rate,
            "epoch_reduction": "mean_of_trajectory_means_also_record_transition_weighted",
            "epoch_prediction_mse": [row["prediction_mse"] for row in epoch_records],
            "epoch_inverse_mse": [row["inverse_mse"] for row in epoch_records],
            "epoch_affordance_mse": [row["affordance_mse"] for row in epoch_records],
            "epochs_detail": epoch_records, "final_diagnostics": self.training_diagnostics(),
            "elapsed_seconds": time.perf_counter() - started,
        }

    @torch.no_grad()
    def training_diagnostics(self, affordance_threshold: float = .5) -> dict[str, Any]:
        """Evaluate all observed transitions and all known-state G outputs."""
        differences = self.Q[:, self._targets] - self.Q[:, self._sources]
        prediction = differences - self.V
        inverse = self.W @ differences
        identity = torch.eye(self.n_edges, device=self.device)
        raw_g = self.G @ self.Q
        truth = self._availability.bool()
        predicted = raw_g >= affordance_threshold
        tp = int((truth & predicted).sum().item())
        fp = int((~truth & predicted).sum().item())
        fn = int((truth & ~predicted).sum().item())
        legal_correct = 0
        for edge, (source, _) in enumerate(self.edges):
            outgoing = self._outgoing[source]
            legal_correct += int(outgoing[int(torch.argmax(inverse[list(outgoing), edge]).item())] == edge)
        separations = torch.pdist(self.Q.T)
        return {
            "prediction_mse": float(prediction.square().mean().item()),
            "prediction_error_l2_mean": float(torch.norm(prediction, dim=0).mean().item()),
            "prediction_error_l2_max": float(torch.norm(prediction, dim=0).max().item()),
            "inverse_mse": float((inverse - identity).square().mean().item()),
            "inverse_global_top1": float((inverse.argmax(dim=0) == torch.arange(self.n_edges, device=self.device)).float().mean().item()),
            "inverse_legal_top1": legal_correct / self.n_edges,
            "affordance_mse": float((raw_g - self._availability).square().mean().item()),
            "affordance_threshold": affordance_threshold, "affordance_true_positive": tp,
            "affordance_false_positive": fp, "affordance_false_negative": fn,
            "affordance_precision": tp / (tp + fp) if tp + fp else 0.,
            "affordance_recall": tp / (tp + fn) if tp + fn else 0.,
            "affordance_empty_states": int((predicted.sum(dim=0) == 0).sum().item()),
            "affordance_raw_min": float(raw_g.min().item()), "affordance_raw_max": float(raw_g.max().item()),
            "state_coordinate_variance": float(self.Q.var(dim=1, unbiased=False).mean().item()),
            "mean_state_norm": float(torch.norm(self.Q, dim=0).mean().item()),
            "mean_action_norm": float(torch.norm(self.V, dim=0).mean().item()),
            "minimum_state_separation": float(separations.min().item()) if len(separations) else None,
            "median_state_separation": float(separations.median().item()) if len(separations) else None,
            "parameter_frobenius_norms": {name: float(torch.norm(getattr(self, name)).item()) for name in ("Q", "V", "W", "G")},
        }

    @torch.no_grad()
    def action_scores(self, state: int, goal: int) -> torch.Tensor:
        scores = self.W @ (self.Q[:, self._state(goal)] - self.Q[:, self._state(state)])
        _finite(scores, "action scores")
        return scores

    @torch.no_grad()
    def greedy_edge_with_diagnostics(
        self, state: int, goal: int, outgoing_edge_indices: Iterable[int], *,
        affordance_mode: str = "observed_graph", affordance_threshold: float = .5,
    ) -> tuple[Optional[int], dict[str, Any]]:
        """Choose the highest W score under an explicit current-action mask.

        The default retains the caller's true outgoing set. In learned mode,
        every action ID with GQ[state] >= threshold is eligible; the supplied
        outgoing set is neither read nor intersected with that prediction.
        Callers must audit an illegal selection without silently repairing it.
        """
        state, goal = self._state(state), self._state(goal)
        if affordance_mode not in {"observed_graph", "learned"}:
            raise ValueError("unknown greedy affordance_mode")
        if affordance_mode == "learned":
            affordance_threshold = _number(affordance_threshold, "affordance_threshold", allow_zero=True)
            if affordance_threshold > 1:
                raise ValueError("affordance_threshold must be <= 1")
            raw_g = self.G @ self.Q[:, state]
            _finite(raw_g, "current-state greedy affordance")
            eligible = (raw_g >= affordance_threshold).nonzero(as_tuple=False).flatten().cpu().tolist()
            scores = self.action_scores(state, goal)[eligible]
            edge = None if state == goal or not eligible else eligible[int(scores.argmax().item())]
            return edge, {
                "selected_edge": edge,
                "branch": "goal_reached" if state == goal else "greedy_learned_g_w" if edge is not None else "no_candidates",
                "candidate_edges": eligible, "w_scores": scores.cpu().tolist(), "negative_scores_allowed": True,
                "model_transitions": 0, "selected_candidate_id": None, "candidates": [],
                "affordance_mode": "learned", "affordance_threshold": affordance_threshold,
                "candidate_affordances": raw_g[eligible].cpu().tolist(),
                "all_action_affordances": raw_g.cpu().tolist(),
                "provided_current_outgoing_used_for_selection": False,
                "oracle_action_intersection": False,
            }
        legal = self._legal(state, outgoing_edge_indices)
        scores = self.action_scores(state, goal)[legal]
        edge = None if state == goal or not legal else legal[int(scores.argmax().item())]
        return edge, {
            "selected_edge": edge, "branch": "goal_reached" if state == goal else "greedy_legal_w" if edge is not None else "no_candidates",
            "candidate_edges": legal, "w_scores": scores.cpu().tolist(), "negative_scores_allowed": True,
            "model_transitions": 0, "selected_candidate_id": None, "candidates": [],
        }

    def _associate(self, latent: torch.Tensor, radius: float, margin: float) -> list[dict[str, Any]]:
        # Matrix distances select the IDs; direct subtraction avoids cancellation
        # in the logged small nearest distances of nearly exact predictions.
        distances = torch.cdist(latent.T, self.Q.T)
        indices = torch.topk(distances, min(2, self.n_states), largest=False, dim=1).indices
        result = []
        for column in range(latent.shape[1]):
            first = int(indices[column, 0].item())
            nearest = float(torch.norm(latent[:, column] - self.Q[:, first]).item())
            second = float(torch.norm(latent[:, column] - self.Q[:, int(indices[column, 1].item())]).item()) if self.n_states > 1 else None
            relative_margin = (second - nearest) / max(second, 1e-12) if second is not None else 1.
            result.append({"node": first, "nearest_distance": nearest, "second_distance": second,
                           "margin": relative_margin, "associated": nearest <= radius and relative_margin >= margin})
        return result

    @torch.no_grad()
    def plan_with_diagnostics(
        self, state: int, goal: int, outgoing_edge_indices: Iterable[int], edge_costs: Sequence[float],
        *, candidates: int = 64, horizon: int = 96, noise_scale: float = .2,
        affordance_threshold: float = .5, association_radius: float = .1,
        association_margin: float = .1, affordance_mode: str = "learned",
        cycle_policy: str = "terminate_on_retrieved_state", proposal: str = "utility",
        soft_temperature: float = .1, soft_weight: float = .5,
        diagnostic_detail: str = "full",
        first_step_affordance_mode: str = "observed_graph",
        latent_projection: str = "none",
        noise_schedule: str = "legacy_active",
    ) -> tuple[Optional[int], dict[str, Any]]:
        """Generate bounded continuous rollouts and execute their best first edge.

        Utilities use an L2 norm over *all* actions before Gaussian noise.
        By default, raw G is used only as a thresholded hard mask, not
        as a probability/multiplier. The first transition always uses the
        caller's true current outgoing set by default. Setting
        first_step_affordance_mode="learned" instead uses GQ[state]'s hard
        threshold over every action ID, without reading the supplied outgoing
        set. Later transitions use learned G
        or explicitly declared observed-graph availability at retrieved Q.
        Optional learned_soft instead permits every later action and adds
        a log-sigmoid G penalty; even true graph sinks cannot abstain.
        Default hard inference and its RNG call sequence remain unchanged.
        Optional indexed_full noise derives an independent per-decision seed
        from the task planning seed and planning-call index, then draws a full
        [depth, candidate, action] table on the model device. Candidate deaths
        do not shift another candidate's noise. Its seed excludes state, goal,
        graph distances, noise scale, and device index.

        Reliable nearest-Q retrieval detects goals/candidate-local cycles.
        The default preserves the continuous latent vector. Optional
        confident_nearest projection replaces z + V by its nearest Q only
        when the existing absolute radius and relative margin both pass;
        rejected retrievals keep their continuous successor. Projection never
        uses an edge's recorded target or graph audit. No complete predicted goal route
        means no_plan. Complete candidates use summed observed edge costs,
        with candidate index as deterministic tie-break; no remaining-goal
        distance or ground-truth continuity audit influences selection.
        """
        started = time.perf_counter()
        state, goal = self._state(state), self._state(goal)
        if first_step_affordance_mode not in {"observed_graph", "learned"}:
            raise ValueError("unknown first_step_affordance_mode")
        if latent_projection not in {"none", "confident_nearest"}:
            raise ValueError("unknown latent_projection")
        if noise_schedule not in {"legacy_active", "indexed_full"}:
            raise ValueError("unknown noise_schedule")
        legal = self._legal(state, outgoing_edge_indices) if first_step_affordance_mode == "observed_graph" else []
        candidates, horizon = _integer(candidates, "candidates", 1), _integer(horizon, "horizon", 1)
        noise_scale = _number(noise_scale, "noise_scale", allow_zero=True)
        affordance_threshold = _number(affordance_threshold, "affordance_threshold", allow_zero=True)
        association_radius = _number(association_radius, "association_radius")
        association_margin = _number(association_margin, "association_margin", allow_zero=True)
        if association_margin > 1 or affordance_threshold > 1:
            raise ValueError("association_margin and affordance_threshold must be <= 1")
        initial_affordance = None
        if first_step_affordance_mode == "learned":
            initial_affordance = self.G @ self.Q[:, state]
            _finite(initial_affordance, "current-state initial affordance")
            legal = (initial_affordance >= affordance_threshold).nonzero(as_tuple=False).flatten().cpu().tolist()
        if affordance_mode not in {"learned", "observed_graph", "learned_soft"}:
            raise ValueError("unknown affordance_mode")
        if diagnostic_detail not in {"full", "compact"}:
            raise ValueError("unknown diagnostic_detail")
        if affordance_mode != "learned_soft" and diagnostic_detail != "full":
            raise ValueError("Hard/observed-graph modes retain their original full diagnostics")
        if affordance_mode == "learned_soft":
            soft_temperature = _number(soft_temperature, "soft_temperature")
            soft_weight = _number(soft_weight, "soft_weight", allow_zero=True)
        if cycle_policy not in {"terminate_on_retrieved_state", "none"} or proposal not in {"utility", "random"}:
            raise ValueError("unknown cycle policy or proposal")
        if len(edge_costs) != self.n_edges:
            raise ValueError("edge_costs must cover every action")
        costs = [_number(value, "edge cost", allow_zero=True) for value in edge_costs]
        self.planning_calls += 1
        indexed_noise = None
        decision_noise_seed = None
        if noise_schedule == "indexed_full":
            # A dedicated generator leaves the historical CPU RNG untouched.
            # Device ordinal and policy choices never enter seed derivation.
            planning_seed = self._planning_generator.initial_seed()
            seed_key = f"cmnav.gcml.indexed_full.v1:{planning_seed}:{self.planning_calls}"
            decision_noise_seed = int.from_bytes(hashlib.sha256(seed_key.encode("ascii")).digest()[:8], "big")
            decision_generator = torch.Generator(device=self.device).manual_seed(decision_noise_seed)
            indexed_noise = torch.randn(
                (horizon, candidates, self.n_edges), generator=decision_generator,
                dtype=torch.float32, device=self.device,
            )
        records: list[dict[str, Any]] = [
            {"id": index, "actions": [], "steps": [], "predicted_arrived": state == goal,
             "termination": "initial_goal" if state == goal else "horizon_exhausted", "cost": 0.}
            for index in range(candidates)
        ]
        result: dict[str, Any] = {
            "selected_edge": None, "branch": "goal_reached" if state == goal else "no_candidates" if not legal else "no_plan",
            "selected_candidate_id": None, "candidate_edges": legal, "candidates": records,
            "planning_call_index": self.planning_calls, "rng_state_before_sha256": _generator_hash(self._planning_generator),
            "model_transitions": 0, "model_transition_budget": candidates * horizon,
            "planning": {"candidates": candidates, "horizon": horizon, "noise_scale": noise_scale,
                         "utility_normalization": "all_actions_l2", "affordance_threshold": affordance_threshold,
                         "association_radius": association_radius, "association_margin": association_margin,
                         "association_distance": "absolute_l2", "association_margin_definition": "(second-first)/second",
                         "affordance_mode": affordance_mode, "cycle_policy": cycle_policy, "proposal": proposal,
                         "first_step_mask": "true_current_outgoing", "latent_update": "continuous_z_plus_v_no_snapping",
                         "selection": "minimum_observed_edge_cost_among_predicted_arrivals", "no_plan_policy": "stop",
                         "ground_truth_candidate_audit_used_for_selection": False},
        }
        if noise_schedule == "indexed_full":
            result["planning"].update({
                "noise_schedule": "indexed_full",
                "decision_noise_seed": decision_noise_seed,
                "noise_base_planning_seed": self._planning_generator.initial_seed(),
                "noise_decision_index": self.planning_calls,
                "noise_seed_derivation": "first_8_bytes_big_endian_sha256_cmnav.gcml.indexed_full.v1:base_seed:decision_index",
                "noise_index_order": ["depth", "candidate_id", "action_id"],
                "noise_table_shape": [horizon, candidates, self.n_edges],
                "noise_table_distribution": "standard_normal_float32_then_scale_at_use",
                "noise_generator_device": str(self.device),
                "noise_legacy_generator_consumed": False,
            })
        if first_step_affordance_mode == "learned":
            result["planning"].update({
                "first_step_affordance_mode": "learned",
                "first_step_mask": "learned_G_threshold_without_oracle_intersection",
                "provided_current_outgoing_used_for_selection": False,
                "first_step_oracle_action_intersection": False,
            })
            result["initial_affordances"] = initial_affordance.cpu().tolist()
            if state != goal and not legal:
                for record in records:
                    record["termination"] = "empty_affordance"
        if latent_projection == "confident_nearest":
            result["planning"].update({
                "latent_projection": "confident_nearest",
                "latent_update": "z_plus_V_then_confident_nearest_Q_projection",
                "projection_confidence": "existing_absolute_L2_radius_and_relative_margin",
                "projection_rejected": "preserve_continuous_successor",
                "logged_nearest_distance": "pre_projection",
                "projection_uses_observed_edge_target": False,
            })
        if affordance_mode == "learned_soft":
            result["planning"].update({
                "soft_gate": {"formula": "u_norm + weight*logsigmoid((Gz-center)/temperature) + Gaussian_noise",
                              "center": affordance_threshold, "temperature": soft_temperature, "weight": soft_weight,
                              "first_step_soft_bonus": False, "future_eligible_scope": "all_action_ids",
                              "future_oracle_filter": False, "learned_abstain": False,
                              "true_graph_sinks_still_propose_actions": True},
                "diagnostic_detail": diagnostic_detail, "soft_compact_topk": 8,
            })
        if state != goal and legal:
            latent = self.Q[:, state, None].repeat(1, candidates)
            active = list(range(candidates))
            associations = [{"node": state, "nearest_distance": 0., "second_distance": None, "margin": 1., "associated": True}
                            for _ in range(candidates)]
            visited = [{state} for _ in range(candidates)]
            for depth in range(horizon):
                if not active:
                    break
                column_ids = torch.tensor(active, dtype=torch.long, device=self.device)
                current_latents = latent[:, column_ids]
                utility = self.W @ (self.Q[:, goal, None] - current_latents)
                # In learned-first mode use the same current-state readout for
                # eligibility and its logs, avoiding batch/vector rounding
                # disagreement at the threshold. The default matmul is intact.
                raw_g = (initial_affordance[:, None].repeat(1, len(active))
                         if depth == 0 and initial_affordance is not None
                         else self.G @ current_latents)
                _finite(utility, "rollout utility")
                _finite(raw_g, "rollout affordance")
                normalized = utility / torch.norm(utility, dim=0, keepdim=True).clamp_min(1e-12)
                if proposal == "random":
                    normalized = torch.zeros_like(normalized)
                noise = (noise_scale * indexed_noise[depth, column_ids, :].T
                         if indexed_noise is not None else
                         noise_scale * torch.randn(utility.shape, generator=self._planning_generator, dtype=torch.float32).to(self.device))
                mask = torch.zeros_like(utility, dtype=torch.bool)
                for local, candidate in enumerate(active):
                    if depth == 0:
                        mask[legal, local] = True
                    elif affordance_mode == "learned":
                        mask[:, local] = raw_g[:, local] >= affordance_threshold
                    elif affordance_mode == "learned_soft":
                        # Pure soft comparison: even a true graph sink has all
                        # actions available. No future graph/abstain filter.
                        mask[:, local] = True
                    elif associations[candidate]["associated"]:
                        mask[list(self._outgoing[associations[candidate]["node"]]), local] = True
                soft_penalty = None
                if affordance_mode == "learned_soft" and depth > 0:
                    soft_penalty = soft_weight * torch.nn.functional.logsigmoid(
                        (raw_g - affordance_threshold) / soft_temperature)
                    _finite(soft_penalty, "soft affordance penalty")
                nonempty = mask.any(dim=0)
                # Empty columns are handled before argmax, never mapped to edge 0.
                next_active, moved_columns, chosen_actions = [], [], []
                for local, candidate in enumerate(active):
                    if not bool(nonempty[local].item()):
                        records[candidate]["termination"] = (
                            "state_unassociated" if affordance_mode == "observed_graph" and not associations[candidate]["associated"]
                            else "empty_affordance")
                        continue
                    if soft_penalty is None:
                        logits = (normalized[:, local] + noise[:, local]).masked_fill(~mask[:, local], -torch.inf)
                    else:
                        logits = (normalized[:, local] + soft_penalty[:, local] + noise[:, local]).masked_fill(~mask[:, local], -torch.inf)
                    action = int(torch.argmax(logits).item())
                    eligible = mask[:, local].nonzero(as_tuple=False).flatten()
                    before = associations[candidate]
                    records[candidate]["actions"].append(action)
                    records[candidate]["cost"] += costs[action]
                    if soft_penalty is not None and diagnostic_detail == "compact":
                        top = torch.topk(logits, min(8, self.n_edges))
                        top_ids = top.indices.cpu().tolist()
                        step_record = {
                            "diagnostic_schema": "cmnav.gcml_soft_step_compact.v1",
                            "index": depth, "action": action, "raw_utility": float(utility[action, local].item()),
                            "normalized_utility": float(normalized[action, local].item()),
                            "noise": float(noise[action, local].item()), "affordance": float(raw_g[action, local].item()),
                            "soft_affordance_bonus": float(soft_penalty[action, local].item()),
                            "logit": float(logits[action].item()), "utility_l2_norm": float(torch.norm(utility[:, local]).item()),
                            "eligible_scope": "all_action_ids", "eligible_edge_id_range": [0, self.n_edges],
                            "eligible_edge_id_range_end_exclusive": True,
                            "eligible_count": self.n_edges, "all_action_scores_stored": False,
                            "global_max_logit": float(logits.max().item()),
                            "selected_rank": 1 + int((logits > logits[action]).sum().item()),
                            "top_by_total_logit": [{"edge": edge, "raw_utility": float(utility[edge, local].item()),
                                "normalized_utility": float(normalized[edge, local].item()),
                                "affordance": float(raw_g[edge, local].item()),
                                "soft_affordance_bonus": float(soft_penalty[edge, local].item()),
                                "noise": float(noise[edge, local].item()), "logit": float(logits[edge].item())}
                                for edge in top_ids],
                            "decoded_state_before": before["node"], "associated_before": before["associated"],
                        }
                    else:
                        step_record = {
                            "index": depth, "action": action, "raw_utility": float(utility[action, local].item()),
                            "normalized_utility": float(normalized[action, local].item()), "noise": float(noise[action, local].item()),
                            "affordance": float(raw_g[action, local].item()), "logit": float(logits[action].item()),
                            "utility_l2_norm": float(torch.norm(utility[:, local]).item()),
                            "eligible_edges": eligible.cpu().tolist(),
                            "eligible_raw_utilities": utility[eligible, local].cpu().tolist(),
                            "eligible_normalized_utilities": normalized[eligible, local].cpu().tolist(),
                            "eligible_affordances": raw_g[eligible, local].cpu().tolist(),
                            "eligible_noise": noise[eligible, local].cpu().tolist(),
                            "decoded_state_before": before["node"], "associated_before": before["associated"],
                        }
                        if soft_penalty is not None:
                            step_record["soft_affordance_bonus"] = float(soft_penalty[action, local].item())
                            step_record["eligible_soft_affordance_bonuses"] = soft_penalty[eligible, local].cpu().tolist()
                    if affordance_mode == "learned_soft" and depth == 0:
                        step_record["soft_affordance_bonus"] = 0.0
                        step_record["soft_bonus_applied"] = False
                    records[candidate]["steps"].append(step_record)
                    next_active.append(candidate)
                    moved_columns.append(local)
                    chosen_actions.append(action)
                if not next_active:
                    break
                action_ids = torch.tensor(chosen_actions, device=self.device)
                successor = current_latents[:, moved_columns] + self.V[:, action_ids]
                _finite(successor, "imagined successor")
                successor_associations = self._associate(successor, association_radius, association_margin)
                result["model_transitions"] += len(next_active)
                continuing = []
                for local, candidate in enumerate(next_active):
                    association = successor_associations[local]
                    snapped = latent_projection == "confident_nearest" and association["associated"]
                    latent[:, candidate] = self.Q[:, association["node"]] if snapped else successor[:, local]
                    associations[candidate] = association
                    arrived = bool(association["associated"] and association["node"] == goal)
                    records[candidate]["steps"][-1].update({
                        "decoded_state_after": association["node"], "nearest_distance": association["nearest_distance"],
                        "second_distance": association["second_distance"], "association_margin": association["margin"],
                        "associated_after": association["associated"], "goal_after": arrived,
                    })
                    if latent_projection == "confident_nearest":
                        records[candidate]["steps"][-1].update({
                            "latent_projection": "confident_nearest", "snapped": bool(snapped),
                            "pre_projection_nearest_distance": association["nearest_distance"],
                            "pre_projection_second_distance": association["second_distance"],
                            "pre_projection_association_margin": association["margin"],
                            "pre_projection_associated": association["associated"],
                            "post_projection_nearest_distance": 0. if snapped else association["nearest_distance"],
                            "projected_node": association["node"] if snapped else None,
                        })
                    if arrived:
                        records[candidate].update(predicted_arrived=True, termination="predicted_goal")
                    elif (cycle_policy == "terminate_on_retrieved_state" and association["associated"]
                          and association["node"] in visited[candidate]):
                        records[candidate]["termination"] = "retrieved_state_cycle"
                    else:
                        if association["associated"]:
                            visited[candidate].add(association["node"])
                        continuing.append(candidate)
                active = continuing
            complete = [record for record in records if record["predicted_arrived"] and record["actions"]]
            if complete:
                best = min(complete, key=lambda record: (record["cost"], record["id"]))
                result.update(selected_edge=best["actions"][0], branch="complete_plan", selected_candidate_id=best["id"])
        result.update(
            rng_state_after_sha256=_generator_hash(self._planning_generator),
            predicted_complete_candidates=sum(record["predicted_arrived"] for record in records),
            distinct_action_sequences=len({tuple(record["actions"]) for record in records}),
            termination_counts={reason: sum(record["termination"] == reason for record in records)
                                for reason in sorted({record["termination"] for record in records})},
            planning_seconds=time.perf_counter() - started,
        )
        return result["selected_edge"], result

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-safe model, vocabulary, protocol and both RNG states."""
        return {
            "schema": _SCHEMA, "algorithm": _ALGORITHM, "backend": "torch", "dtype": "float32",
            "reference_metadata": self.reference_metadata, "n_states": self.n_states,
            "edges": [list(edge) for edge in self.edges], "dimension": self.dimension, "seed": self.seed,
            "learning_rate": self.learning_rate, "state_learning_rate": self.state_learning_rate,
            "inverse_learning_rate": self.inverse_learning_rate, "affordance_learning_rate": self.affordance_learning_rate,
            "readout_step_rule": _STEP_RULE, "readout_step_safety": self.readout_step_safety,
            "updates": self.updates, "trajectory_updates": self.trajectory_updates,
            "epochs_completed": self.epochs_completed, "planning_calls": self.planning_calls,
            **{name: getattr(self, name).detach().cpu().tolist() for name in ("Q", "V", "W", "G")},
            "shuffle_generator_state": self._shuffle_generator.get_state().tolist(),
            "planning_generator_state": self._planning_generator.get_state().tolist(),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any], *, device: str | torch.device = "cpu") -> "GenerativeCognitiveMapLearner":
        if (data.get("schema") != _SCHEMA or data.get("algorithm") != _ALGORITHM or data.get("backend") != "torch"
                or data.get("dtype") != "float32" or data.get("readout_step_rule") != _STEP_RULE):
            raise ValueError("unsupported GCML schema, algorithm, backend or readout step rule")
        model = cls(
            n_states=data["n_states"], edges=data["edges"], dimension=data["dimension"], seed=data["seed"],
            learning_rate=data["learning_rate"], state_learning_rate=data["state_learning_rate"],
            inverse_learning_rate=data["inverse_learning_rate"], affordance_learning_rate=data["affordance_learning_rate"],
            readout_step_safety=data["readout_step_safety"], reference_metadata=data["reference_metadata"], device=device,
        )
        for name, shape in (("Q", (model.dimension, model.n_states)), ("V", (model.dimension, model.n_edges)),
                            ("W", (model.n_edges, model.dimension)), ("G", (model.n_edges, model.dimension))):
            tensor = torch.tensor(data[name], dtype=torch.float32, device=model.device)
            if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"invalid GCML {name} shape or nonfinite entries")
            setattr(model, name, tensor)
        for name in ("updates", "trajectory_updates", "epochs_completed", "planning_calls"):
            setattr(model, name, _integer(data[name], name))
        for name in ("shuffle", "planning"):
            values = data[f"{name}_generator_state"]
            if not isinstance(values, list) or not values:
                raise ValueError(f"invalid GCML {name} RNG state")
            values = [_integer(value, "generator state byte") for value in values]
            if any(value > 255 for value in values):
                raise ValueError("invalid GCML generator state byte")
            try:
                getattr(model, f"_{name}_generator").set_state(torch.tensor(values, dtype=torch.uint8))
            except RuntimeError as error:
                raise ValueError(f"invalid GCML {name} RNG state") from error
        return model

"""CML trajectory updates and goal-direction action selection in PyTorch.

Training follows ``train_model`` in the authors' reference notebook at
https://github.com/IGITUGraz/Cognitive-Map-Learner/blob/
e36a17771bbe758c1d6b5f0446aa4a167b154a12/Cognitive%20Map%20Learner.ipynb.
Each trajectory forms one batch. Its prediction errors are computed before
any writes, including the reference's indexed Q/V writes for repeated IDs.

Each directed edge has its own action vector. Inputs are discrete state IDs,
not images or Habitat motor actions. Action availability remains a hard
candidate restriction, and the W-history/V-fallback selection rule is retained.
The module performs no graph search, simulator query, or GCML imagination.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader


_SCHEMA = "cmnav.cml.v2"
_ALGORITHM = "cml_2024_torch_trajectory_batches"
_LEGACY_SCHEMA = "cmnav.cml.v1"
_LEGACY_ALGORITHM = "cml_2024_sequential_directed_edges"
_REFERENCE_COMMIT = "e36a17771bbe758c1d6b5f0446aa4a167b154a12"
_REPEATED_INDEX_POLICY = "official_advanced_indexed_write_not_accumulation"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _rate(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"nonfinite CML {name}")


class CognitiveMapLearner:
    """Learn Q/V prediction embeddings and W's Hebbian readout without RL.

    Q=(D,N), V=(D,E), W=(E,D), with Gaussian initialization scales 1, .1,
    and .1. New models use float32. Initialization and trajectory shuffling
    have separate, privately seeded generators; neither uses global RNG state.
    CPU is the default device, with CUDA available through an explicit device.

    A batch computes ``d = Q[:, targets] - Q[:, sources]`` and
    ``error = d - V[:, actions]`` from pre-update values. Q and V then use
    the official advanced-indexed assignments, while W sums Hebbian terms.
    Repeated Q/V indices are deliberately not replaced by ``index_add_`` or
    a sequential loop. These writes inherit PyTorch's backend behavior; the
    seed does not promise identical results across devices or Torch versions.

    Historical v1 snapshots retain float64 tensors and their original schema.
    They support inference and reserialization, but cannot resume training.
    """

    def __init__(
        self,
        n_states: int,
        edges: Sequence[tuple[int, int]],
        dimension: int = 1000,
        seed: int = 0,
        learning_rate: float = 0.01,
        *,
        state_learning_rate: float = 0.1,
        inverse_learning_rate: float = 0.01,
        normalize_actions: bool = False,
        device: str | torch.device = "cpu",
    ):
        self.n_states = _integer(n_states, "n_states", 1)
        self.dimension = _integer(dimension, "dimension", 1)
        self.seed = _integer(seed, "seed")
        if self.seed > 2**64 - 1:
            raise ValueError("seed must fit an unsigned 64-bit integer")
        self.learning_rate = _rate(learning_rate, "learning_rate")
        self.state_learning_rate = _rate(state_learning_rate, "state_learning_rate")
        self.inverse_learning_rate = _rate(inverse_learning_rate, "inverse_learning_rate")
        if not isinstance(normalize_actions, bool):
            raise ValueError("normalize_actions must be a boolean")
        self.normalize_actions = normalize_actions
        self.edges: tuple[tuple[int, int], ...] = tuple(
            self._parse_edge(edge) for edge in edges
        )
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("each directed (source, target) edge must have one action ID")
        self.n_edges = len(self.edges)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("CML device must be cpu or cuda")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        tensor_options = {"dtype": torch.float32, "device": self.device, "generator": generator}
        self.Q = torch.randn((self.dimension, self.n_states), **tensor_options)
        self.V = 0.1 * torch.randn((self.dimension, self.n_edges), **tensor_options)
        self.W = 0.1 * torch.randn((self.n_edges, self.dimension), **tensor_options)
        self.device = self.Q.device
        self._shuffle_generator = torch.Generator(device="cpu")
        self._shuffle_generator.manual_seed(self.seed)
        self._schema = _SCHEMA
        self._algorithm = _ALGORITHM
        self.read_only = False
        self.updates = 0
        self.trajectory_updates = 0
        self.epochs_completed = 0

    def _state(self, value: Any, name: str = "state") -> int:
        result = _integer(value, name)
        if result >= self.n_states:
            raise ValueError(f"{name} is outside the state vocabulary")
        return result

    def _parse_edge(self, edge: Any) -> tuple[int, int]:
        if len(edge) != 2:
            raise ValueError("each edge must contain source and target state IDs")
        return self._state(edge[0], "source"), self._state(edge[1], "target")

    def _edge_id(self, value: Any) -> int:
        result = _integer(value, "edge ID")
        if result >= self.n_edges:
            raise ValueError("edge ID is outside the action vocabulary")
        return result

    def _trajectories(
        self, trajectories: Iterable[Sequence[tuple[int, int, int]]]
    ) -> list[torch.Tensor]:
        records = []
        for trajectory in trajectories:
            triples = []
            previous_target = None
            for transition in trajectory:
                if len(transition) != 3:
                    raise ValueError("each transition must contain (source, edge ID, target)")
                source = self._state(transition[0], "source")
                action = self._edge_id(transition[1])
                target = self._state(transition[2], "target")
                if self.edges[action] != (source, target):
                    raise ValueError("transition does not match its directed edge ID")
                if previous_target is not None and source != previous_target:
                    raise ValueError("trajectory transitions must be continuous; separate resets")
                triples.append((source, action, target))
                previous_target = target
            if not triples:
                raise ValueError("fit requires nonempty trajectories")
            records.append(torch.tensor(triples, dtype=torch.long, device="cpu"))
        if not records:
            raise ValueError("fit requires at least one trajectory")
        return records

    @torch.no_grad()
    def fit(
        self,
        trajectories: Iterable[Sequence[tuple[int, int, int]]],
        epochs: int = 10,
    ) -> dict[str, Any]:
        """Train on complete trajectories, shuffling their order each epoch.

        Trajectories contain ``(source, edge_id, target)`` triples and may have
        different lengths. Batch size is one trajectory, not one transition.
        The caller must keep reset boundaries separate and remove empty paths.
        ``updates`` counts consumed transitions; ``trajectory_updates`` counts
        actual batches. Optional normalization covers every V column after
        each batch. No optimizer, autograd, or learning-rate schedule is used.
        """
        if self.read_only:
            raise RuntimeError("historical CML v1 models are inference-only; create a new v2 model to train")
        epochs = _integer(epochs, "epochs", 1)
        records = self._trajectories(trajectories)
        loader = DataLoader(
            records, batch_size=1, shuffle=True, num_workers=0,
            generator=self._shuffle_generator,
        )
        transitions_per_epoch = sum(len(record) for record in records)
        epoch_mse: list[float] = []
        trajectory_mse: list[float] = []
        initial_updates = self.updates
        initial_trajectory_updates = self.trajectory_updates
        for _ in range(epochs):
            mean_mse = 0.0
            for count, trajectory in enumerate(loader, start=1):
                trajectory = trajectory[0].to(self.device)
                sources, actions, targets = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
                state_diff = self.Q[:, targets] - self.Q[:, sources]
                prediction_error = state_diff - self.V[:, actions]
                desired = torch.nn.functional.one_hot(actions, num_classes=self.n_edges).T.to(self.Q.dtype)
                _require_finite(state_diff, "batch state differences")
                _require_finite(prediction_error, "batch prediction errors")
                # Preserve the notebook's indexed-write semantics for repeated IDs.
                self.Q[:, targets] += -self.state_learning_rate * prediction_error
                self.V[:, actions] += self.learning_rate * prediction_error
                self.W += self.inverse_learning_rate * desired @ state_diff.T
                if self.normalize_actions:
                    norms = torch.norm(self.V, dim=0)
                    _require_finite(norms, "action norms")
                    if bool((norms == 0).any().item()):
                        raise FloatingPointError("CML action normalization has a zero norm")
                    self.V = self.V / norms
                for name in ("Q", "V", "W"):
                    _require_finite(getattr(self, name), f"{name} after batch update")
                mse = float(torch.mean(prediction_error.square()).item())
                if not math.isfinite(mse):
                    raise FloatingPointError("nonfinite CML batch prediction MSE")
                self.updates += len(trajectory)
                self.trajectory_updates += 1
                trajectory_mse.append(mse)
                mean_mse += (mse - mean_mse) / count
            epoch_mse.append(mean_mse)
            self.epochs_completed += 1
        return {
            "algorithm": self._algorithm,
            "backend": "torch",
            "dtype": "float32",
            "device": str(self.Q.device),
            "epochs": epochs,
            "trajectories_per_epoch": len(records),
            "transitions_per_epoch": transitions_per_epoch,
            "updates_added": self.updates - initial_updates,
            "updates_total": self.updates,
            "update_count_unit": "consumed_transitions",
            "trajectory_updates_added": self.trajectory_updates - initial_trajectory_updates,
            "trajectory_updates_total": self.trajectory_updates,
            "repeated_index_policy": _REPEATED_INDEX_POLICY,
            "trajectory_prediction_mse": trajectory_mse,
            "epoch_prediction_mse": epoch_mse,
            "epoch_prediction_mse_reduction": "mean_of_trajectory_means",
            "state_coordinate_variance": float(torch.var(self.Q, dim=1, unbiased=False).mean().item()),
            "mean_state_norm": float(torch.norm(self.Q, dim=0).mean().item()),
            "mean_action_norm": float(torch.norm(self.V, dim=0).mean().item()),
        }

    @torch.no_grad()
    def action_scores(self, state: int, goal: int) -> torch.Tensor:
        """Return W utilities for all edges on the model's device.

        These geometric scores are not calibrated probabilities and do not
        include availability or history masking.
        """
        state, goal = self._state(state), self._state(goal, "goal")
        scores = self.W @ (self.Q[:, goal] - self.Q[:, state])
        _require_finite(scores, "action utilities")
        return scores

    def choose_edge(
        self,
        state: int,
        goal: int,
        outgoing_edge_indices: Iterable[int],
        used_edges: Iterable[int] = (),
    ) -> Optional[int]:
        """Choose an unused positive-W edge, otherwise the best legal V edge.

        V fallback permits revisits. Ties select the smallest edge ID; at the
        goal or with no legal outgoing edge, return None. No graph search or
        cost weighting is introduced by the PyTorch training change.
        """
        edge, _ = self.choose_edge_with_diagnostics(state, goal, outgoing_edge_indices, used_edges)
        return edge

    @torch.no_grad()
    def choose_edge_with_diagnostics(
        self,
        state: int,
        goal: int,
        outgoing_edge_indices: Iterable[int],
        used_edges: Iterable[int] = (),
    ) -> tuple[Optional[int], dict[str, Any]]:
        """Return the chosen edge and JSON-safe scores actually considered.

        An empty score list means that branch was not evaluated. The caller
        owns action history; this method neither mutates it nor the model.
        """
        state, goal = self._state(state), self._state(goal, "goal")
        candidates = sorted({self._edge_id(value) for value in outgoing_edge_indices})
        if any(self.edges[edge][0] != state for edge in candidates):
            raise ValueError("provided outgoing edge has a different source state")
        used = {self._edge_id(value) for value in used_edges}
        unused = [edge for edge in candidates if edge not in used]
        diagnostics: dict[str, Any] = {
            "selected_edge": None,
            "branch": "goal_reached" if state == goal else "no_candidates",
            "candidate_edges": candidates,
            "used_candidate_edges": [edge for edge in candidates if edge in used],
            "w_candidate_edges": [],
            "w_scores": [],
            "w_positive_threshold": 0.0,
            "v_candidate_edges": [],
            "v_scores": [],
        }
        if state == goal or not candidates:
            return None, diagnostics
        direction = self.Q[:, goal] - self.Q[:, state]
        _require_finite(direction, "goal direction")
        if unused:
            scores = self.W[unused] @ direction
            _require_finite(scores, "candidate utilities")
            diagnostics["w_candidate_edges"] = unused
            diagnostics["w_scores"] = scores.cpu().tolist()
            best_index = int(torch.argmax(scores).item())
            if float(scores[best_index].item()) > 0:
                edge = unused[best_index]
                diagnostics.update(selected_edge=edge, branch="w_unused")
                return edge, diagnostics
        fallback = self.V[:, candidates].T @ direction
        _require_finite(fallback, "fallback utilities")
        edge = candidates[int(torch.argmax(fallback).item())]
        diagnostics.update(
            selected_edge=edge, branch="v_fallback", v_candidate_edges=candidates,
            v_scores=fallback.cpu().tolist(),
        )
        return edge, diagnostics

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-safe weights and, for v2, the shuffle generator state.

        Legacy snapshots retain their original schema, algorithm and dtype;
        their stored numerical values are never converted to float32.
        """
        result = {
            "schema": self._schema,
            "algorithm": self._algorithm,
            "reference_commit": _REFERENCE_COMMIT,
            "n_states": self.n_states,
            "edges": [list(edge) for edge in self.edges],
            "dimension": self.dimension,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "state_learning_rate": self.state_learning_rate,
            "inverse_learning_rate": self.inverse_learning_rate,
            "normalize_actions": self.normalize_actions,
            "dtype": "float64" if self.read_only else "float32",
            "updates": self.updates,
            "epochs_completed": self.epochs_completed,
            "Q": self.Q.detach().cpu().tolist(),
            "V": self.V.detach().cpu().tolist(),
            "W": self.W.detach().cpu().tolist(),
        }
        if not self.read_only:
            result.update(
                backend="torch",
                trajectory_updates=self.trajectory_updates,
                repeated_index_policy=_REPEATED_INDEX_POLICY,
                shuffle_generator_state=self._shuffle_generator.get_state().tolist(),
            )
        return result

    @classmethod
    def from_state_dict(
        cls, data: Mapping[str, Any], *, device: str | torch.device = "cpu"
    ) -> "CognitiveMapLearner":
        """Load v2 for training/inference or v1 for float64 inference only.

        The execution device is explicit, not restored from an old machine.
        Tensor reductions may differ from the former NumPy backend even for
        unchanged v1 weights; historical decisions are not claimed bit-exact.
        """
        legacy = data.get("schema") == _LEGACY_SCHEMA
        expected_algorithm = _LEGACY_ALGORITHM if legacy else _ALGORITHM
        expected_dtype = "float64" if legacy else "float32"
        if data.get("schema") not in {_SCHEMA, _LEGACY_SCHEMA} or data.get("algorithm") != expected_algorithm:
            raise ValueError("unsupported CML model schema or algorithm")
        if data.get("reference_commit") != _REFERENCE_COMMIT or data.get("dtype") != expected_dtype:
            raise ValueError("unsupported CML model reference or dtype")
        if not legacy and (
            data.get("backend") != "torch"
            or data.get("repeated_index_policy") != _REPEATED_INDEX_POLICY
        ):
            raise ValueError("unsupported CML training backend or repeated-index policy")
        model = cls(
            n_states=data["n_states"],
            edges=data["edges"],
            dimension=data["dimension"],
            seed=data["seed"],
            learning_rate=data["learning_rate"],
            state_learning_rate=data["state_learning_rate"],
            inverse_learning_rate=data["inverse_learning_rate"],
            normalize_actions=data["normalize_actions"],
            device=device,
        )
        dtype = torch.float64 if legacy else torch.float32
        for name, shape in (
            ("Q", (model.dimension, model.n_states)),
            ("V", (model.dimension, model.n_edges)),
            ("W", (model.n_edges, model.dimension)),
        ):
            tensor = torch.tensor(data[name], dtype=dtype, device=model.device)
            # JSON cannot represent the trailing dimension of an empty matrix.
            if name == "W" and model.n_edges == 0 and tensor.shape == (0,):
                tensor = tensor.reshape(shape)
            if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"invalid CML {name} shape or nonfinite values")
            setattr(model, name, tensor)
        model.updates = _integer(data["updates"], "updates")
        model.epochs_completed = _integer(data["epochs_completed"], "epochs_completed")
        model._schema = data["schema"]
        model._algorithm = data["algorithm"]
        model.read_only = legacy
        if not legacy:
            model.trajectory_updates = _integer(data["trajectory_updates"], "trajectory_updates")
            generator_state = data["shuffle_generator_state"]
            if not isinstance(generator_state, list) or not generator_state:
                raise ValueError("invalid CML shuffle generator state")
            values = [_integer(value, "shuffle generator byte") for value in generator_state]
            if any(value > 255 for value in values):
                raise ValueError("invalid CML shuffle generator byte")
            try:
                model._shuffle_generator.set_state(torch.tensor(values, dtype=torch.uint8, device="cpu"))
            except RuntimeError as error:
                raise ValueError("invalid CML shuffle generator state") from error
        return model

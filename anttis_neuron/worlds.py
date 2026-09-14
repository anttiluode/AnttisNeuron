"""Deterministic branched-cable worlds for Gate 6 robustness tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .branch import spectral_radius
from .cable import cable_operator, visible_physical_modes


@dataclass(frozen=True, eq=False)
class CableWorld:
    index: int
    seed: int
    n_nodes: int
    edges: np.ndarray
    port_nodes: tuple[int, int, int, int]
    sensor_nodes: tuple[int, int, int, int]
    adaptation_angles: tuple[float, float, float, float]
    heldout_angles: tuple[float, float, float, float, float]
    adaptation_spectra: tuple[tuple[float, float, float, float], ...]
    heldout_spectrum: tuple[float, float, float, float]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CableWorld):
            return NotImplemented
        return (
            self.index == other.index
            and self.seed == other.seed
            and self.n_nodes == other.n_nodes
            and np.array_equal(self.edges, other.edges)
            and self.port_nodes == other.port_nodes
            and self.sensor_nodes == other.sensor_nodes
            and self.adaptation_angles == other.adaptation_angles
            and self.heldout_angles == other.heldout_angles
            and self.adaptation_spectra == other.adaptation_spectra
            and self.heldout_spectrum == other.heldout_spectrum
        )


def port_sensor_matrices(world: CableWorld) -> tuple[np.ndarray, np.ndarray]:
    """Build one-hot four-port input and four-sensor observation matrices."""
    b = np.zeros((world.n_nodes, 4), dtype=float)
    s = np.zeros((4, world.n_nodes), dtype=float)
    for column, node in enumerate(world.port_nodes):
        b[node, column] = 1.0
    for row, node in enumerate(world.sensor_nodes):
        s[row, node] = 1.0
    return b, s


def _random_parent_tree(rng: np.random.Generator, n_nodes: int) -> np.ndarray:
    edges = np.empty((n_nodes - 1, 2), dtype=int)
    for node in range(1, n_nodes):
        parent = int(rng.integers(0, node))
        edges[node - 1] = (parent, node)
    return edges


def _degrees(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    degree = np.zeros(n_nodes, dtype=int)
    for i, j in edges:
        degree[int(i)] += 1
        degree[int(j)] += 1
    return degree


def _jitter_spectrum(rng: np.random.Generator, base: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    factors = rng.uniform(0.85, 1.15, size=4)
    values = np.asarray(base, dtype=float) * factors
    return tuple(float(v) for v in values)


def generate_world(index: int, master_seed: int = 1701, *, max_attempts: int = 64) -> CableWorld:
    """Generate one deterministic valid 11-node world.

    Every attempt is seeded from `(master_seed, index, attempt)`. Invalid
    observer geometries are rejected deterministically; callers therefore get
    the same accepted world for the same inputs on every run.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    n_nodes = 11
    base_adaptation_angles = np.asarray((12.0, 31.0, 53.0, 74.0), dtype=float)
    base_heldout_angles = np.asarray((22.0, 42.0, 63.0, 83.0, 103.0), dtype=float)
    base_adaptation_spectra = (
        (2.6, 0.8, 1.2, 0.6),
        (1.8, 0.7, 2.1, 0.5),
        (2.6, 0.8, 1.2, 0.6),
        (1.8, 0.7, 2.1, 0.5),
    )
    base_heldout_spectrum = (2.2, 0.9, 1.6, 0.55)

    for attempt in range(max_attempts):
        sequence = np.random.SeedSequence([master_seed, index, attempt])
        world_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(sequence)

        edges = _random_parent_tree(rng, n_nodes)
        degree = _degrees(n_nodes, edges)
        leaves = np.flatnonzero((degree == 1) & (np.arange(n_nodes) != 0))
        candidates = leaves if len(leaves) >= 4 else np.arange(1, n_nodes)
        port_nodes = tuple(int(v) for v in rng.choice(candidates, size=4, replace=False))
        sensor_nodes = tuple(int(v) for v in rng.choice(np.arange(1, n_nodes), size=4, replace=False))

        adaptation_angles_array = base_adaptation_angles + rng.uniform(-12.0, 12.0, size=4)
        heldout_angles_array = base_heldout_angles + rng.uniform(-12.0, 12.0, size=5)
        for h in range(len(heldout_angles_array)):
            if np.any(np.abs(adaptation_angles_array - heldout_angles_array[h]) <= 1e-6):
                heldout_angles_array[h] += 0.5

        adaptation_spectra = tuple(
            _jitter_spectrum(rng, spectrum) for spectrum in base_adaptation_spectra
        )
        heldout_spectrum = _jitter_spectrum(rng, base_heldout_spectrum)

        world = CableWorld(
            index=index,
            seed=world_seed,
            n_nodes=n_nodes,
            edges=edges,
            port_nodes=port_nodes,
            sensor_nodes=sensor_nodes,
            adaptation_angles=tuple(float(v) for v in adaptation_angles_array),
            heldout_angles=tuple(float(v) for v in heldout_angles_array),
            adaptation_spectra=adaptation_spectra,
            heldout_spectrum=heldout_spectrum,
        )
        b, s = port_sensor_matrices(world)
        del b  # stability depends only on the physical operator; visibility uses S.
        try:
            a = cable_operator(
                n_nodes,
                edges,
                np.ones(len(edges), dtype=float),
                dt=0.08,
                leak=0.08,
                coupling=0.65,
            )
        except ValueError:
            continue
        if spectral_radius(a) >= 1.0:
            continue
        if visible_physical_modes(a, s).shape[0] < 3:
            continue
        return world

    raise RuntimeError(
        f"failed to generate a valid cable world for index={index} "
        f"after {max_attempts} deterministic attempts"
    )

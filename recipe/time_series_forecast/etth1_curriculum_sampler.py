from __future__ import annotations

import random
from collections.abc import Iterator, Sized

from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractCurriculumSampler


DEFAULT_PHASE_STEPS = (4, 8, 8)
DEFAULT_PHASE_BAND_WEIGHTS = (
    (0.75, 0.20, 0.05),
    (0.40, 0.25, 0.35),
    (0.25, 0.20, 0.55),
)
SUPPORTED_BANDS = (1, 2, 3)


def _allocate_counts(total: int, ratios: tuple[float, ...]) -> list[int]:
    if total <= 0:
        return [0 for _ in ratios]

    raw_counts = [total * ratio / sum(ratios) for ratio in ratios]
    counts = [int(count) for count in raw_counts]
    remainder = total - sum(counts)

    if remainder > 0:
        order = sorted(range(len(ratios)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
        for idx in order[:remainder]:
            counts[idx] += 1

    return counts


class ETTH1BandCurriculumSampler(AbstractCurriculumSampler):
    """Curriculum sampler for the unique ETTH1 BANDDROP training set."""

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        self.data_source = data_source
        self.data_config = data_config

        assert not data_config.get("shuffle", True), "ETTH1BandCurriculumSampler requires data.shuffle=False"
        assert data_config.get("dataloader_num_workers", 0) == 0, (
            "ETTH1BandCurriculumSampler requires data.dataloader_num_workers=0"
        )

        sampler_cfg = data_config.get("sampler", {})
        self.batch_size = int(data_config.get("gen_batch_size") or data_config.get("train_batch_size"))
        self.phase_steps = tuple(int(step) for step in sampler_cfg.get("phase_steps", DEFAULT_PHASE_STEPS))
        self.phase_band_weights = tuple(
            tuple(float(weight) for weight in weights)
            for weights in sampler_cfg.get("phase_band_weights", DEFAULT_PHASE_BAND_WEIGHTS)
        )
        self.seed = int(sampler_cfg.get("seed", data_config.get("seed", 0) or 0))

        if len(self.phase_steps) != len(self.phase_band_weights):
            raise ValueError("phase_steps and phase_band_weights must have the same length")
        if any(len(weights) != len(SUPPORTED_BANDS) for weights in self.phase_band_weights):
            raise ValueError(f"Each phase must provide {len(SUPPORTED_BANDS)} band weights")

        self.rng = random.Random(self.seed)
        self.current_step = 0
        self.band_members = self._build_band_members()
        self.band_orders = {band: members[:] for band, members in self.band_members.items()}
        self.band_positions = {band: 0 for band in SUPPORTED_BANDS}

        for band in SUPPORTED_BANDS:
            self.rng.shuffle(self.band_orders[band])

        self.phase_batch_counts = [
            self._normalize_phase_batch_counts(
                dict(zip(SUPPORTED_BANDS, _allocate_counts(self.batch_size, weights), strict=False)),
                weights,
            )
            for weights in self.phase_band_weights
        ]

    def __iter__(self) -> Iterator[int]:
        phase_counts = self.phase_batch_counts[self._phase_index()]
        batch_indices: list[int] = []
        batch_seen: set[int] = set()

        for band in SUPPORTED_BANDS:
            batch_indices.extend(self._take_from_band(band=band, count=phase_counts[band], batch_seen=batch_seen))

        self.rng.shuffle(batch_indices)
        return iter(batch_indices)

    def __len__(self) -> int:
        return self.batch_size

    def update(self, batch) -> None:
        self.current_step += 1

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "band_orders": {band: order[:] for band, order in self.band_orders.items()},
            "band_positions": dict(self.band_positions),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.current_step = int(state_dict["current_step"])
        self.band_orders = {int(band): list(order) for band, order in state_dict["band_orders"].items()}
        self.band_positions = {int(band): int(pos) for band, pos in state_dict["band_positions"].items()}
        self.rng.setstate(state_dict["rng_state"])

    def _phase_index(self) -> int:
        remaining_steps = self.current_step
        for phase_index, phase_steps in enumerate(self.phase_steps):
            if remaining_steps < phase_steps:
                return phase_index
            remaining_steps -= phase_steps
        return len(self.phase_steps) - 1

    def _build_band_members(self) -> dict[int, list[int]]:
        band_members = {band: [] for band in SUPPORTED_BANDS}
        for index in range(len(self.data_source)):
            row = self._get_row(index)
            extra_info = row.get("extra_info", {}) if isinstance(row, dict) else {}
            band = int(extra_info["band"])
            if band in band_members:
                band_members[band].append(index)

        missing = [band for band, members in band_members.items() if not members]
        if missing:
            raise ValueError(f"Missing ETTH1 band members for bands: {missing}")

        return band_members

    def _get_row(self, index: int):
        dataframe = getattr(self.data_source, "dataframe", None)
        if dataframe is not None:
            try:
                return dataframe[index]
            except Exception:
                pass
        return self.data_source[index]

    def _take_from_band(self, band: int, count: int, batch_seen: set[int]) -> list[int]:
        selected: list[int] = []
        selected_set: set[int] = set()

        while len(selected) < count:
            order = self.band_orders[band]
            position = self.band_positions[band]

            if position >= len(order):
                self.band_orders[band] = self.band_members[band][:]
                self.rng.shuffle(self.band_orders[band])
                self.band_positions[band] = 0
                order = self.band_orders[band]
                position = 0

            candidate = order[position]
            self.band_positions[band] = position + 1
            if candidate in selected_set or candidate in batch_seen:
                continue

            selected.append(candidate)
            selected_set.add(candidate)
            batch_seen.add(candidate)

        return selected

    def _normalize_phase_batch_counts(
        self,
        counts: dict[int, int],
        weights: tuple[float, ...],
    ) -> dict[int, int]:
        counts = dict(counts)
        capacities = {band: len(self.band_members[band]) for band in SUPPORTED_BANDS}
        overflow = 0

        for band in SUPPORTED_BANDS:
            if counts[band] > capacities[band]:
                overflow += counts[band] - capacities[band]
                counts[band] = capacities[band]

        while overflow > 0:
            eligible_bands = [band for band in SUPPORTED_BANDS if counts[band] < capacities[band]]
            if not eligible_bands:
                raise ValueError("No capacity left to redistribute curriculum counts without duplicates")

            eligible_weights = tuple(weights[SUPPORTED_BANDS.index(band)] for band in eligible_bands)
            additions = _allocate_counts(overflow, eligible_weights)

            assigned = 0
            for band, addition in zip(eligible_bands, additions, strict=False):
                room = capacities[band] - counts[band]
                actual_addition = min(addition, room)
                counts[band] += actual_addition
                assigned += actual_addition

            overflow -= assigned

        return counts

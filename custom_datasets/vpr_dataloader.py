# samplers_views_v2.py
import numpy as np
from typing import List, Tuple
import torch


RGB, THR = 0, 1

class IntraDatasetViewBatchSamplerV2:
    """
    GLOBAL-index sampler for a Concat-like MultiDatasetWrapper where each index has both RGB & Thermal views.
    Emits a list of (global_idx, view_id) pairs per batch.

    Per anchor a (global index):
      - Always include (a, RGB) and (a, THR)
      - Add up to k_hard_pos hard positives (indices != a), one view each
      - Add up to k_soft_pos soft positives (indices != a & not already picked), one view each
      - Add neg_pool ring negatives (indices != a), one view each

    Epoch handling (finite):
      - steps_per_epoch controls how many batches per epoch
      - __len__() == steps_per_epoch
      - call set_epoch(epoch) each epoch (important for DDP determinism)

    NOTE: batch_samplers run in the MAIN process. Worker-aware seeding must be done via DataLoader's worker_init_fn
    for dataset/augmentation randomness. This sampler is epoch- and rank-aware (DDP-safe).
    """

    def __init__(
        self,
        wrapper,
        anchors_per_batch: int = 32,
        k_hard_pos: int = 1,
        k_soft_pos: int = 1,
        neg_pool: int = 16,
        steps_per_epoch: int = 1000,
        dataset_mix: dict = None,  # e.g. {"DatasetA": 0.5, "DatasetB": 0.5} or None for equal mix
        seed: int = 0,
        pos_view_policy: str = "random",   # "random" or "balanced"
        neg_view_policy: str = "balanced"  # "random" or "balanced"
    ):
        self.w = wrapper
        self.ds_list = list(wrapper.datasets)

        # Concat-like global index ranges per underlying dataset
        if hasattr(wrapper, "cumulative_sizes") and len(wrapper.cumulative_sizes) == len(self.ds_list):
            cs = np.asarray(wrapper.cumulative_sizes, dtype=np.int64)
            starts = np.zeros_like(cs); starts[1:] = cs[:-1]
            self.starts = starts
        else:
            lens = np.array([len(ds) for ds in self.ds_list], dtype=np.int64)
            starts = np.zeros_like(lens); starts[1:] = np.cumsum(lens)[:-1]
            self.starts = starts
        self.ends = self.starts + np.array([len(ds) for ds in self.ds_list], dtype=np.int64)

        # Params
        self.A = int(max(1, anchors_per_batch))
        self.k_hard_pos = int(max(0, k_hard_pos))
        self.k_soft_pos  = int(max(0, k_soft_pos))
        self.neg_pool = int(max(0, neg_pool))
        self.steps_per_epoch = int(max(1, steps_per_epoch))
        self.base_seed = int(seed)

        # dataset sampling probabilities
        if dataset_mix is None:
            probs = np.ones(len(self.ds_list), dtype=np.float64)
        else:
            probs = []
            for ds in self.ds_list:
                name = getattr(ds, "name", None) or ds.__class__.__name__
                probs.append(float(dataset_mix.get(name, 0.0)))
            probs = np.asarray(probs, dtype=np.float64)
            if probs.sum() <= 0:
                probs[:] = 1.0
        self.ds_probs = probs / probs.sum()

        # required wrapper methods
        assert hasattr(wrapper, "hard_positives_per_query")
        assert hasattr(wrapper, "extra_margin_soft_positives")
        assert hasattr(wrapper, "ring_negatives")

        assert pos_view_policy in ("random", "balanced")
        assert neg_view_policy in ("random", "balanced")
        self.pos_view_policy = pos_view_policy
        self.neg_view_policy = neg_view_policy

        # epoch state
        self._epoch = 0

    # ---- utilities ----
    def _range(self, ds_id: int) -> Tuple[int, int]:
        return int(self.starts[ds_id]), int(self.ends[ds_id])

    def _hard(self, gidx: int) -> List[int]:
        if len(self.w.hard_positives_per_query) == 0:
            return []
        else:
            return self.w.hard_positives_per_query[gidx]

    def _soft(self, gidx: int) -> List[int]:
        # return list(self.w.extra_margin_soft_positives[gidx] or [])
        if len(self.w.extra_margin_soft_positives) == 0:
            return []
        else:
            return self.w.extra_margin_soft_positives[gidx]
        

    def _ring(self, gidx: int) -> List[int]:
        if len(self.w.ring_negatives) == 0:
            return []
        else:
            return self.w.ring_negatives[gidx]

    def _sample_anchor_from_ds(self, ds_id: int, rng: np.random.Generator, require_pos: bool = True) -> int:
        lo, hi = self._range(ds_id)
        for _ in range(64):
            a = int(rng.integers(lo, hi))
            if not require_pos:
                return a
            if (len(self._hard(a)) > 0) or (len(self._soft(a)) > 0):
                return a
        return None

    def _pick_views(self, count: int, start_toggle: int, policy: str, rng: np.random.Generator) -> List[int]:
        if count <= 0:
            return []
        if policy == "balanced":
            out, toggle = [], start_toggle
            for _ in range(count):
                out.append(toggle)
                toggle = THR if toggle == RGB else RGB
            return out
        # random
        return [int(rng.integers(0, 2)) for _ in range(count)]

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch (important for DDP determinism)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    # ---- iterator with epoch+rank aware seeding ----
    def __iter__(self):
        # rank-aware seed (DDP safe)
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0

        # single RNG for this epoch (main process; batch_sampler is not in workers)
        seed = self.base_seed + self._epoch * 1000 + rank * 100_000
        rng = np.random.default_rng(seed)

        steps = 0
        while steps < self.steps_per_epoch:
            # choose ONE dataset for this batch (intra-dataset)
            ds_id = int(rng.choice(len(self.ds_list), p=self.ds_probs))

            # pick anchors
            anchors: List[int] = []
            tries = 0
            while len(anchors) < self.A and tries < 20 * self.A:
                tries += 1
                a = self._sample_anchor_from_ds(ds_id, rng, require_pos=True)
                if a is not None:
                    anchors.append(a)

            if not anchors:
                break  # end epoch early if nothing valid

            # build (idx, view) pairs
            batch: List[Tuple[int, int]] = []
            pos_view_toggle = RGB
            neg_view_toggle = RGB

            for a in anchors:
                # (1) cross-modal at same index
                batch.append((a, RGB))
                batch.append((a, THR))

                # (2) extra positives
                hardP_all = [x for x in self._hard(a) if x != a]
                softP_all = [x for x in self._soft(a) if x != a]
                hard_set = set(hardP_all)
                softP = [x for x in softP_all if x not in hard_set]
                hardP = hardP_all

                hard_sel = []
                if self.k_hard_pos > 0 and hardP:
                    take_h = min(self.k_hard_pos, len(hardP))
                    hard_sel = list(rng.choice(hardP, size=take_h, replace=False))

                soft_sel = []
                if self.k_soft_pos > 0 and softP:
                    pool = [x for x in softP if x not in hard_sel]
                    if pool:
                        take_s = min(self.k_soft_pos, len(pool))
                        soft_sel = list(rng.choice(pool, size=take_s, replace=False))

                pos_indices = hard_sel + soft_sel
                if pos_indices:
                    pos_views = self._pick_views(len(pos_indices), pos_view_toggle, self.pos_view_policy, rng)
                    for idx, v in zip(pos_indices, pos_views):
                        batch.append((int(idx), int(v)))
                    if self.pos_view_policy == "balanced" and (len(pos_indices) % 2 == 1):
                        pos_view_toggle = THR if pos_view_toggle == RGB else RGB

                # (3) ring negatives
                neg_candidates = [x for x in self._ring(a) if x != a and x not in pos_indices]
                if self.neg_pool > 0 and neg_candidates:
                    take_n = min(self.neg_pool, len(neg_candidates))
                    neg_idx = list(rng.choice(neg_candidates, size=take_n, replace=False))
                    if self.neg_view_policy == "balanced":
                        for n in neg_idx:
                            batch.append((int(n), neg_view_toggle))
                            neg_view_toggle = THR if neg_view_toggle == RGB else RGB
                    else:
                        neg_views = self._pick_views(len(neg_idx), THR, "random", rng)
                        for n, v in zip(neg_idx, neg_views):
                            batch.append((int(n), int(v)))

            yield batch
            steps += 1


class ViewIndexingDataset(torch.utils.data.Dataset):
    """
    Adapts your MultiDatasetWrapper (Concat-like) so that __getitem__ accepts (idx, view) tuples.
    Expects the wrapper's __getitem__(idx) to return both modalities, e.g.:
      sample = wrapper[idx]  -> dict like {"rgb": tensor, "thr": tensor, ...}
    Returns a dict with the chosen 'image' and some meta for mining.
    """
    def __init__(self, wrapper, rgb_key="rgb", thr_key="thr"):
        self.w = wrapper
        self.rgb_key = rgb_key
        self.thr_key = thr_key

    def __len__(self):
        return sum(len(ds) for ds in self.w.datasets)

    def __getitem__(self, key):
        # key is either an int (fallback) or a (idx, view) tuple
        if isinstance(key, (list, tuple)) and len(key) == 2:
            idx, view = int(key[0]), int(key[1])
        else:
            idx, view = int(key), RGB  # default to RGB if view is not provided

        sample = self.w[idx]["item"][0]  # should return both modalities
        
        if isinstance(sample, dict):
            rgb = sample[self.rgb_key]
            thr = sample[self.thr_key]
        else:
            raise ValueError(f"Expected sample to be a dict with keys '{self.rgb_key}' and '{self.thr_key}', got {type(sample)}")

        img = rgb if view == RGB else thr
        return {
            "image": img,                 # (C,H,W)
            "base_idx": torch.tensor(idx, dtype=torch.long),
            "view_id":  torch.tensor(view, dtype=torch.long),
            "rgb": rgb,                   # keep both around if you like
            "thr": thr,
            # Add other fields from 'sample' if your pipeline needs them
        }
    def get_hard_positives_per_query(self):
        return self.w.hard_positives_per_query
    def get_extra_margin_soft_positives(self):
        return self.w.extra_margin_soft_positives
    def get_ring_negatives(self):
        return self.w.ring_negatives

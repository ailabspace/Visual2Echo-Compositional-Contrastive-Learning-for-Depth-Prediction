"""
Wraps AudioVisualDataset to inject precomputed teacher features per sample.

The HDF5 cache stores rows indexed by the same integer used by DataLoader's
sampler (i.e. dataset[idx]). Both the base dataset and the cache are accessed
with the same idx, so shuffle is transparent — no ordering assumptions needed.
"""

import torch
import h5py
import torch.utils.data as data


class CachedLatentDataset(data.Dataset):
    """
    Thin wrapper around an AudioVisualDataset that injects cached teacher
    latents (img_feat, material_feat, material_class) into each sample dict.

    Args:
        base_dataset: an initialized AudioVisualDataset instance.
        cache_path:   path to the HDF5 file written by precompute_teacher_latents.py.
    """

    def __init__(self, base_dataset, cache_path):
        self.base = base_dataset
        # Open once; h5py handles thread safety via per-process file handles.
        # Use swmr=False (default) since we only read.
        self._cache_path = cache_path
        self._cache = None  # opened lazily in each worker via __getitem__

    def _open_cache(self):
        # Lazy open so that the file handle is created in the DataLoader worker
        # process, not the main process (avoids fork-related h5py issues).
        if self._cache is None:
            self._cache = h5py.File(self._cache_path, 'r')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        self._open_cache()
        item = self.base[idx]
        if item is None:
            return None

        # Load fp16 from HDF5, upcast to fp32 for training
        if 'img_depth' in self._cache:
            item['img_depth'] = torch.from_numpy(
                self._cache['img_depth'][idx]).float()
        item['img_feat']        = torch.from_numpy(
            self._cache['img_feat'][idx]).float()
        item['material_feat']   = torch.from_numpy(
            self._cache['material_feat'][idx]).float()
        item['material_class_teacher'] = torch.from_numpy(
            self._cache['material_class'][idx]).float()

        return item

    def name(self):
        return 'CachedLatentDataset'

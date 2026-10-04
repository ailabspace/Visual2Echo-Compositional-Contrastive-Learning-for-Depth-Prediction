import h5py
import torch
import torch.nn.functional as F
import torch.utils.data as data


class CachedLatentDataset(data.Dataset):
    def __init__(self, base_dataset, cache_path, material_cache_path=None):
        self.base = base_dataset
        self._cache_path = cache_path
        self._cache = None
        self._mat_cache_path = material_cache_path or None
        self._mat_cache = None
        if self._mat_cache_path:
            with h5py.File(self._mat_cache_path, 'r') as m, h5py.File(cache_path, 'r') as t:
                assert len(m['material_class']) == len(t['material_class']), 'material/teacher cache row mismatch'
        self._target_hw = tuple(getattr(base_dataset, 'target_hw', ()) or ())

    def _to_target(self, t):
        if not self._target_hw or t.dim() != 3 or tuple(t.shape[-2:]) == self._target_hw:
            return t
        return F.interpolate(t.unsqueeze(0).float(), size=self._target_hw, mode='bilinear', align_corners=False)[0]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self._cache is None:
            self._cache = h5py.File(self._cache_path, 'r')
        if self._mat_cache_path and self._mat_cache is None:
            self._mat_cache = h5py.File(self._mat_cache_path, 'r')
        item = self.base[idx]
        c = self._cache
        if 'img_depth' in c:
            item['img_depth'] = self._to_target(torch.from_numpy(c['img_depth'][idx]).float())
        for k in ('enc_feat', 'enc_feat_multi', 'img_feat'):
            if k in c:
                item[k] = torch.from_numpy(c[k][idx]).float()
                break
        mc = self._mat_cache if self._mat_cache is not None else c
        item['material_feat'] = torch.from_numpy(mc['material_feat'][idx]).float()
        item['material_class_teacher'] = torch.from_numpy(mc['material_class'][idx]).float()
        return item

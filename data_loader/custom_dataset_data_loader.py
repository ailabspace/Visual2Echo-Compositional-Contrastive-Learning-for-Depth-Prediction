import torch
import torch.utils.data


def CreateDataset(opt):
    if getattr(opt, 'dataset', '') == 'biosonar':
        from data_loader.biosonar_dataset import BiosonarDataset
        dataset = BiosonarDataset()
    else:
        from data_loader.audio_visual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader():
    def __init__(self, mode="base"):
        self.mode = mode

    def name(self):
        return 'CustomDatasetDataLoader'

    def custom_collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) <= 1:
            return None
        return {
            key: torch.stack([
                d[key] if isinstance(d[key], torch.Tensor) else torch.tensor(d[key])
                for d in batch
            ])
            for key in batch[0].keys()
        }

    def initialize(self, opt, teacher_cache_path=None, material_cache_path=None):
        self.dataset = CreateDataset(opt)

        if teacher_cache_path is not None:
            from data_loader.cached_latent_dataset import CachedLatentDataset
            self.dataset = CachedLatentDataset(self.dataset, teacher_cache_path,
                                               material_cache_path=material_cache_path)
            print(f'[DataLoader] Using cached teacher latents: {teacher_cache_path}')

        shuff = opt.mode == "train"
        _nw = int(opt.nThreads)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=shuff,
            num_workers=_nw,
            collate_fn=self.custom_collate_fn,
            drop_last=True,
            pin_memory=_nw > 0,
            persistent_workers=_nw > 0,
            prefetch_factor=2 if _nw > 0 else None,
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataloader:
            yield data

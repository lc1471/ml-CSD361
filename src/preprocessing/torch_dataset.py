"""Thin PyTorch wrapper around ClipReader. Needs torch; run where torch works (Colab / cluster)."""
import os, sys
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ClipReader  # noqa: E402


class AVCAffeDataset(Dataset):
    def __init__(self, root, split, **kwargs):
        self.reader = ClipReader(root, split, **kwargs)

    def __len__(self):
        return len(self.reader)

    def set_epoch(self, epoch):
        self.reader.set_epoch(epoch)

    def __getitem__(self, i):
        d = self.reader[i]
        return {"video": torch.from_numpy(d["video"]), "audio": torch.from_numpy(d["audio"]),
                "labels": torch.from_numpy(d["labels"]), "has_audio": torch.tensor(d["has_audio"]),
                "clip_id": d["clip_id"]}


def collate_all_windows(batch):
    """For mode='all': flatten every clip's windows into one batch; labels/has_audio repeat per window."""
    counts = [b["video"].shape[0] for b in batch]
    return {"video": torch.cat([b["video"] for b in batch]),
            "audio": torch.cat([b["audio"] for b in batch]),
            "labels": torch.cat([b["labels"].unsqueeze(0).expand(c, -1) for b, c in zip(batch, counts)]),
            "has_audio": torch.cat([b["has_audio"].reshape(1).expand(c) for b, c in zip(batch, counts)]),
            "clip_id": [b["clip_id"] for b, c in zip(batch, counts) for _ in range(c)]}

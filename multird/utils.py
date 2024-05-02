import json
from datetime import datetime
from typing import NamedTuple
from typing import Any
from pathlib import Path

import torch
import numpy as np

class Trajectory(NamedTuple):

    loss: list[float]
    accuracy: list[dict[int, float]]

    @staticmethod
    def create() -> 'Trajectory':
        return Trajectory([], [])


class History(NamedTuple):

    train: Trajectory
    valid: Trajectory

    @staticmethod
    def create() -> 'History':
        return History(Trajectory.create(), Trajectory.create())

    def to_dict(self):
        return {
            "train": self.train._asdict(),
            "valid": self.valid._asdict(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> 'History':
        return History(**dict((k, Trajectory(**v)) for k, v in d.items()))



class Restoration(NamedTuple):

    state_dict: dict[str, Any]
    epoch: int
    history: History



class Checkpointer:

    def __init__(self,
                 checkpoint_dir: Path,
                 history: History,
                 mode: str):
        self.checkpoint_dir = checkpoint_dir
        self.history = history
        self.mode = mode

    def save_results(self, index2word: np.ndarray, label_list: list[int], pred_list: list[list[int]], epoch: int):
        saved_files = []
        for infix, content in {
            "label": (index2word[label_list]).tolist(),
            "pred": (index2word[np.array(pred_list)]).tolist()
        }.items():
            list_file = self.checkpoint_dir / f"checkpoint-epoch{epoch:02d}-{self.mode}_{infix}_list.json"
            with open(list_file, "w") as ofile:
                json.dump(content, ofile, indent=2)
            saved_files.append(list_file)
        return saved_files

    @staticmethod
    def restore(checkpoint_file: Path, pt_device: str = None) -> 'Restoration':
        checkpoint = torch.load(str(checkpoint_file), map_location=pt_device)
        history = History.from_dict(checkpoint["history"])
        checkpoint["history"] = history
        checkpoint = dict((k, v) for k, v in checkpoint.items() if k in Restoration._fields)
        return Restoration(**checkpoint)

    def checkpoint(self, model: torch.nn.Module, epoch: int) -> Path:
        saved_files = {}
        epoch_infix = f"epoch{epoch:02d}"
        checkpoint_file = self.checkpoint_dir / f"model-{epoch_infix}.pt"
        checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "history": self.history.to_dict(),
        }
        torch.save(checkpoint, str(checkpoint_file))
        saved_files["checkpoint"] = checkpoint_file
        return checkpoint_file


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")

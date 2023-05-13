import json
import torch
from pathlib import Path
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from datetime import datetime as dt

class Serializer:

    experiment_result_dir = Path("models_saved/")

    def __init__(self, experiment_name: str, disableb: bool) -> None:

        self.disabled = disableb
        if disableb:
            return
        # Create if not already present common folders
        if not Serializer.experiment_result_dir.exists():
            Serializer.experiment_result_dir.mkdir()

        self.exp_name = experiment_name
        self.exp_dir_path = Serializer._get_experiment_dir_full_path(experiment_name)
        
        if not self.exp_dir_path.exists():
            self.exp_dir_path.mkdir()
        elif not self.exp_dir_path.is_dir():
            raise Exception("The name chosen for the experiment directory is already used by another experiment")

    def save_model(self, model: _SimpleSegmentationModel) -> None:
        if self.disabled:
            return
        model_path = self._get_model_path()
        torch.save(model.state_dict(), model_path)

    def save_params(self, params: dict) -> None:
        if self.disabled:
            return
        self._save_json("params", params)
        
    def save_results(self, params: dict) -> None:
        if self.disabled:
            return
        self._save_json("results", params)

    def _save_json(self, file_name: str, params: dict) -> None:
        file_path = self.exp_dir_path.joinpath(f"{file_name}.json")
        with open(file_path, "a") as outfile:
            json.dump(params, outfile)

    def _get_model_path(self) -> Path:
        return self.exp_dir_path.joinpath(f"model.torch")

    @staticmethod
    def _get_experiment_dir_full_path(experiment_name: str) -> Path:
        time = dt.now().isoformat(timespec="minutes")
        return Serializer.experiment_result_dir.joinpath(f"{time}-{experiment_name}")
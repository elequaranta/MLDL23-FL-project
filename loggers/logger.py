from argparse import Namespace
from datetime import datetime as dt
import enum
import json
from zoneinfo import ZoneInfo
import torch
import wandb
import inspect
from abc import ABC, abstractmethod
from overrides import overrides
from typing import Any, Dict, Literal, Optional
from pathlib import Path
from experiment.snapshot import Snapshot


class Logger(ABC, object):

    def __init__(self, args: Namespace) -> None:
        self._args = args

    @property
    def args(self) -> Namespace:
        return self._args

    @abstractmethod
    def log(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def save(self, snapshot: Snapshot) -> None:
        pass

    @abstractmethod
    def summary(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def watch(self, 
              models: Any, 
              criterion: Any | None = None, 
              log: Literal['gradients', 'parameters', 'all'] | None = "gradients",
              log_freq: int = 1000) -> None:
        pass

    @abstractmethod
    def save_results(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def restore_snapshot(self, file_name: str, run_path: str) -> Optional[Snapshot]:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass


class DummyLogger(Logger):

    @overrides
    def log(self, data: Dict[str, Any]) -> None:
        pass

    @overrides
    def save(self, snapshot: Snapshot) -> None:
        pass

    @overrides
    def summary(self, data: Dict[str, Any]) -> None:
        pass

    @overrides
    def watch(self, 
              models: Any, 
              criterion: Any | None = None, 
              log: Literal['gradients', 'parameters', 'all'] | None = "gradients",
              log_freq: int = 1000) -> None:
        pass

    @overrides
    def save_results(self, data: Dict[str, Any]) -> None:
        pass

    @overrides
    def restore_snapshot(self, file_name: str, run_path: str) -> Optional[Snapshot]:
        return None

    @overrides
    def finish(self) -> None:
        pass

class BaseDecorator(Logger):

    _logger : Logger = None

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    @property
    def logger(self) -> Logger:
        return self._logger
    
    @property
    def args(self) -> Namespace:
        return self._logger.args
    
    @overrides
    def log(self, data: Dict[str, Any]) -> None:
        self._logger.log(data)

    @overrides
    def save(self, snapshot: Snapshot) -> None:
        self._logger.save(snapshot)

    @overrides
    def summary(self, data: Dict[str, Any]) -> None:
        self._logger.summary(data)

    @overrides
    def watch(self, 
              models: Any, 
              criterion: Any | None = None, 
              log: Literal['gradients', 'parameters', 'all'] | None = "gradients",
              log_freq: int = 1000) -> None:
        self._logger.watch(models, criterion, log, log_freq)

    @overrides
    def save_results(self, data: Dict[str, Any]) -> None:
        self._logger.save_results(data)

    @overrides
    def restore_snapshot(self, file_name: str, run_path: str) -> Optional[Snapshot]:
        return self._logger.restore_snapshot(file_name, run_path)

    @overrides
    def finish(self) -> None:
        self._logger.finish()

class WandbLoggerDecorator(BaseDecorator):

    root = Path("wandb_checkpoint")

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        wandb.init(
                    project=self.args.project, 
                    name=self.args.exp_name, 
                    config=vars(self.args)
        )
        if not WandbLoggerDecorator.root.exists():
            WandbLoggerDecorator.root.mkdir()
    
    @overrides
    def log(self, data: Dict[str, Any]) -> None:
        step_value = data.pop("step")
        wandb.log(data=data, step=step_value)
        self.logger.log(data)

    @overrides
    def save(self, snapshot: Snapshot) -> None:
        snapshot_tmp_path = Path(wandb.run.dir).joinpath(f"{snapshot.get_name()}.torch")
        torch.save(snapshot, snapshot_tmp_path)
        wandb.save(snapshot_tmp_path.as_posix(), policy="now")
        self.logger.save(snapshot)

    @overrides
    def summary(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            wandb.summary[k] = v
        self.logger.summary(data)

    @overrides
    def watch(self, 
              models: Any, 
              criterion: Any | None = None, 
              log: Literal['gradients', 'parameters', 'all'] | None = "gradients",
              log_freq: int = 1000) -> None:
        wandb.watch(models, criterion, log, log_freq)
        self._logger.watch(models, criterion, log, log_freq)

    @overrides
    def restore_snapshot(self, file_name: str, run_path: str) -> Optional[Snapshot]:
        file = wandb.restore(file_name, run_path, root=WandbLoggerDecorator.root)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.load(file.name, weights_only=False, map_location=device)

    @overrides
    def finish(self) -> None:
        wandb.finish()
        self.logger.finish()

class LocalLoggerDecorator(BaseDecorator):

    root = Path("models_saved/")

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        if not LocalLoggerDecorator.root.exists():
            LocalLoggerDecorator.root.mkdir()
        self.exp_dir = self._create_dir()
        self._save_params(vars(self.args))

    @overrides
    def save(self, snapshot: Snapshot) -> None:
        snapshot_tmp_path = self.exp_dir.joinpath(f"{snapshot.get_name()}.torch")
        torch.save(snapshot, snapshot_tmp_path)
        self.logger.save(snapshot)

    @overrides
    def save_results(self, data: Dict[str, Any]) -> None:
        file_path = self.exp_dir.joinpath(f"result.json")
        with open(file_path, "a") as outfile:
            json.dump(data, outfile)
        self.logger.save_results(data)

    def _create_dir(self) -> Path:
        timezone = ZoneInfo("Europe/Rome")
        time = dt.now(timezone).isoformat(timespec="minutes")
        exp_dir = LocalLoggerDecorator.root.joinpath(f"{time}-{self.args.exp_name}")
        if not exp_dir.exists():
            exp_dir.mkdir()
        elif not exp_dir.is_dir():
            raise Exception("LocalLogger: Already exist a folder with the same name")
        return exp_dir

    def _save_params(self, data: Dict[str, Any]) -> None:
        file_path = self.exp_dir.joinpath(f"params.json")
        with open(file_path, "a") as outfile:
            json.dump(data, outfile)

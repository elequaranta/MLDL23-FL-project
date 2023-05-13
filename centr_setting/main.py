from argparse import Namespace
from typing import Any, Dict, List
import wandb
from torchvision.datasets import VisionDataset

from centr_setting.centralized_model import CentralizedModel
from utils.init_fs import Serializer
from utils.stream_metrics import StreamSegMetrics


def main(
        args: Namespace, 
        train_datasets: List[VisionDataset], 
        test_datasets: Dict[str, VisionDataset], 
        model: Any, 
        metrics: Dict[str, StreamSegMetrics], 
        serializer: Serializer):
    centralized_model = CentralizedModel(args, train_datasets[0], test_datasets, model, serializer)
    centralized_model.train()
    centralized_model.test(metrics["eval_train"], "train")
    centralized_model.test(metrics["test_same_dom"], "same_dom")
    centralized_model.test(metrics["test_diff_dom"], "diff_dom")
    for key, metric in metrics.items():
        print(f"Test on dataset: {key} => {metric}")
    if not args.not_use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
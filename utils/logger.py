from pathlib import Path
import time
import uuid
import os
from typing import Optional

from omegaconf import DictConfig, OmegaConf
import wandb





class WandBLogger:

    def __init__(self, cfg: DictConfig, experiment_name: str) -> None:
        """
        Initialize the WandB logger.

        Args:
            cfg (DictConfig): The configuration dictionary.
        """
        # WandB
        experiment_path = Path(cfg.output_path) / cfg.model.model / experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)

        os.environ["WANDB_CACHE_DIR"] = os.path.join(
            experiment_path, "wandb", ".cache", "wanb"
        )

        os.environ["WANDB_CAPTURE_STDOUT"] = "false"
        os.environ["WANDB_CAPTURE_STDERR"] = "false"

        print("Cache dir:", os.environ["WANDB_CACHE_DIR"])
        self.logger = wandb.init(
            project=cfg.logging.project,
            name=f"{cfg.model.model}-{experiment_name}",
            dir=experiment_path,
            reinit=True,
            tags=[cfg.model.tag],
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.logging.mode,
            sync_tensorboard=False
        )

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """
        Log metrics to WandB.

        Args:
            metrics (dict): The metrics to log.
            step (int): The step number.
        """
        self.logger.log(metrics, step=step)
        print(f"Epoch {step} logged.")

    def log_artifact(self, artifact_path: str) -> None:
        """
        Log an artifact to WandB.

        Args:
            artifact_path (str): The path to the artifact.
        """
        self.logger.log_artifact(artifact_path)

    def finish(self) -> None:
        """
        Finish the logging process.
        """
        self.logger.finish()
    


def set_logger_paths(cfg: DictConfig) -> tuple[Path, str]:
    """
    Set the paths for the logger.

    Args:
        cfg (DictConfig): The configuration dictionary.
    
    Returns:
        tuple[Path, str]: The experiment path and name. We use the name as an identifier for the experiment, for the WandB logger.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{str(timestr)}_{uuid.uuid4().hex[:5]}"
    experiment_path = Path(cfg['output_path']) / cfg.model.model / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    return experiment_path, experiment_name
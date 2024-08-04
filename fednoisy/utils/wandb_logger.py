import wandb

from fedlab.utils.logger import Logger

from typing import Dict

class WandbLogger():
    def __init__(self, project_name: str, exp_cfg: Dict, *args, **kwargs) -> None:
        self.run = wandb.init(
            project=project_name,
            config=exp_cfg,
            group=kwargs.get("group", None),
            tags=kwargs.get("tags", None),
            job_type=kwargs.get("job_type", None),
            name=kwargs.get("name", None),
        )
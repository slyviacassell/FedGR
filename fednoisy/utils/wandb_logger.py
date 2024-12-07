import wandb

from fedlab.utils.logger import Logger

from typing import Dict

class WandbLogger():
    def __init__(self, project_name: str, exp_cfg: Dict, *args, **kwargs) -> None:
        # if not wandb.login():
        wandb.login(
            key="local-f28a6751c8d295f6b01752a1de7fc84fd768b839",
            host="http://192.168.0.142:8080",
            relogin=True,
        )
        self.run = wandb.init(
            project=project_name,
            config=exp_cfg,
            group=kwargs.get("group", None),
            tags=kwargs.get("tags", None),
            job_type=kwargs.get("job_type", None),
            name=kwargs.get("name", None),
        )
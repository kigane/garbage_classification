import wandb

from utils import read_yaml

#----------------------------------------------------------------------------

def log_dataset(proj_name) -> None:
    """在wandb上记录数据集"""
    with wandb.init(project=proj_name, job_type="log_dataset") as run:
        raw_data = wandb.Artifact(
            "garbage-raw",
            type="dataset",
            description="Raw garbage dataset",
            metadata={
                "classes": "cardboard,glass,metal,paper,plastic,trash",
                "train": 2024,
                "test": 503,
            })
        raw_data.add_dir('dataset')
        run.log_artifact(raw_data)


def log_model(opt) -> None:
    """在wandb上记录模型"""
    model_artifact = wandb.Artifact(
        name=opt.model_name,  # 名称
        type=opt.model_type,  # 类别，分组
        description=opt.model_desc,
        metadata=opt.model_metadata
    )
    model_artifact.add_file('./results/models/' + opt.model_name + '.pth')
    wandb.log_artifact(model_artifact)

#----------------------------------------------------------------------------

class Visualizer():
    def __init__(self, sweep: bool):
        proj_info = read_yaml('project.yml')

        if not sweep:
            config = read_yaml('config.yml')
            wandb.init(project=proj_info['project'],
                    group=proj_info['group'],
                    job_type=proj_info['job_type'],
                    config=config)
        else:
            wandb.init()

        self.opt = wandb.config
        self.opt.update(proj_info)
        
    def add_scalars(self, losses: dict, epoch: int):
        wandb.log(losses, step=epoch)
    
    def add_summary(self, key, val):
        wandb.summary[key] = val

#----------------------------------------------------------------------------

import wandb

#----------------------------------------------------------------------------

def log_dataset(proj_name) -> None:
    """在wandb上记录数据集"""
    with wandb.init(project=proj_name, job_type="log-data") as run:
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


def log_model(info) -> None:
    """在wandb上记录模型"""
    model_artifact = wandb.Artifact(
        name=info.model_name,  # 名称
        type=info.model_type,  # 类别
        description=info.model_desc,
        metadata=info.model_metadata.to_dict()
    )
    model_artifact.add_file('./results/models/' + info.model_name + '.pth')
    wandb.log_artifact(model_artifact)

#----------------------------------------------------------------------------

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.use_wandb = opt.use_wandb
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project='GarbageClassification', name=opt.name, config=opt) if not wandb.run else wandb.run

    def plot_current_losses(self, epoch, losses):
        self.wandb_run.log(losses, step=epoch)

#----------------------------------------------------------------------------

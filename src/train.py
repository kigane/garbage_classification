import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import garbage_dataset
from model import GarbageModel, evaluate_accuracy
from utils import read_yaml, save_model
from visualizer import log_model


def build_model_path(name: str) -> str:
    return 'results/models/' + name + '.pth'


def train(config=None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None:
        wandb.init()
    else:
        wandb.init(project=config.project,
                   group=config.group,
                   job_type=config.job_type,
                   config=config.to_dict())
    opt = wandb.config
    # 准备数据集
    training_set = garbage_dataset('dataset/train')
    valid_set = garbage_dataset('dataset/valid')
    train_iter = DataLoader(
        training_set, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    valid_iter = DataLoader(
        valid_set, batch_size=opt.batch_size, pin_memory=True)
  
    model = GarbageModel(opt, device)

    best_acc = 0
    for epoch in range(opt.n_epochs):
        model.reset_metrics()
        for batch_idx, (X, y) in enumerate(tqdm(train_iter, desc=f'epoch {epoch+1:>3}')):
            # forward
            model.set_input((X, y))
            model.forward()
            # optimize_params
            model.optimize_parameters()
            # record
            model.accumulate_metrics()

        # after train loop
        wandb.log({'train/loss': model.calc_loss()}, step=epoch+1)
        wandb.log({'train/acc': model.calc_acc()}, step=epoch+1)

        valid_acc = evaluate_accuracy(model.net, valid_iter, device)
        wandb.log({'valid/acc': valid_acc}, step=epoch+1)

        if best_acc < valid_acc:
            best_acc = valid_acc
            wandb.summary['best_acc'] = best_acc
            save_model(model, best_acc, build_model_path(opt.model_name))

    # after training
    if config is not None:
        log_model(opt)


if __name__ == '__main__':
    # single run
    config = read_yaml('options.yml')
    train(config)

    # sweep
    # sweep_config = read_yaml('sweep.yml', True)
    # sweep_id = wandb.sweep(sweep_config, project=_PROJECT)
    # wandb.agent(sweep_id, function=train, count=8)

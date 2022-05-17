import torch
import wandb
from tqdm import tqdm

from dataset import get_dataloader
from model import GarbageModel, evaluate_accuracy
from utils import read_yaml
from visualizer import Visualizer, log_model

#----------------------------------------------------------------------------

PROJECT_NAME = "GarbageClassification"

#----------------------------------------------------------------------------

def train(sweep=True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualizer = Visualizer(sweep)
    opt = visualizer.opt
    train_iter, valid_iter = get_dataloader(opt)
    model = GarbageModel(opt, device)

    best_acc = 0
    for epoch in range(opt.n_epochs):
        model.reset_metrics_net()
        for batch_idx, (X, y) in enumerate(tqdm(train_iter, desc=f'epoch {epoch+1:>3}, lr={model.optimizer.param_groups[0]["lr"]}')):
            # train
            model.set_input((X, y))
            model.optimize_parameters()
            # record
            model.accumulate_metrics()

        # after train loop
        valid_acc = evaluate_accuracy(model.net, valid_iter, device)
        visualizer.add_scalars({
            'train/loss': model.calc_loss(),
            'train/acc': model.calc_acc(),
            'valid/acc': valid_acc
        }, step=epoch+1)

        if best_acc < valid_acc:
            best_acc = valid_acc
            visualizer.add_summary('best_acc', best_acc)
            model.save(best_acc)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    # single run
    train(sweep=False)

    # sweep
    # sweep_config = read_yaml('sweep.yml')
    # sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    # wandb.agent(sweep_id, function=train, count=3)

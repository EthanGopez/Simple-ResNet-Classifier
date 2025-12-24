import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


from .load import load_data, TumorImageDataset
from .model import ResNet

def train(
    exp_dir: str = "logs",
    num_epoch: int = 25,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    **kwargs
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
        pin_memory = False
    
    # let's set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    
    log_dir = Path("model/" + exp_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"resnet_{datetime.now().strftime('%m%d_%H%M%S')}.th"

    # define the model
    model = ResNet(**kwargs)
    model = model.to(device)

    # load the generated dataset
    train_data = load_data(annotations_file="data/labels/labels_train.csv",img_dir="data/train", shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_data = load_data(annotations_file="data/labels/labels_val.csv",img_dir="data/val", shuffle=False, pin_memory=pin_memory)

    # hyperparameters of training function
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    grayscale_transform = transforms.Grayscale(num_output_channels=1)

    best_val_acc = 0.0

    for epoch in range(num_epoch):
        metrics = {"train_acc": [], "val_acc": []}
        # ensure model is in training mode for this epoch
        model.train()
        for step, (img, label) in enumerate(train_data):
            # apply grayscale conversion and move tensors to device
            img = img.to(device)

            label = label.to(device).long().view(-1)

            optimizer.zero_grad()

            outputs = model(img)

            # compute loss and backprop
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()

            # calculate train accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == label).float().mean().item()
            metrics["train_acc"].append(acc)

            
        # run validation
        model.eval()
        with torch.inference_mode():
            for img, label in val_data:
                img = grayscale_transform(img)
                img = img.to(device)
                label = label.to(device).long().view(-1)

                # compute validation accuracy
                outputs = model(img)
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == label).float().mean().item()
                metrics["val_acc"].append(acc)
        
        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_acc={epoch_val_acc:.4f}"
        )
        
        # for early stopping
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # save the model under unified name
            torch.save(model.state_dict(), "model/resnet.th")

            # save past name to a log directory
            torch.save(model.state_dict(), log_path)
            print(f"Model from epoch {str(epoch)} saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0) # num processes for train set

    """
        Optional args for model definition and easy hyperparamter tuning

        Args:
            h: height of image
            w: width of image
            out_classes: number of classes
            num_blocks: number of residual blocks for skipping
            depth_blocks: number of hidden layers in each residual block
            hidden_dim: dimensions of output of residual blocks
            bias: learning an additive bias?
    """
    parser.add_argument("--h", type=int, default=28)
    parser.add_argument("--w", type=int, default=28)
    parser.add_argument("--out_classes", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--depth_blocks", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--bias", type=bool, default=False)
    # pass all arguments to train
    train(**vars(parser.parse_args()))
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support

from cifar10 import (
        CIFAR10Multilabel, 
        CIFAR10Hierarchical, 
        cifar10_transform
    )

def train_log(save_path, content, also_print=True):
    if also_print:
        print(content)
    with open(save_path/'train_log.txt', 'a') as f:
        f.write(content + '\n')

def sigmoid(x):
        return np.exp(-np.logaddexp(0, -x))

def evaluate(dataloader, model, threshold=0.5, multilabel=False):

    match multilabel:
        case True:
            loss_func = F.binary_cross_entropy_with_logits
            pred_func = lambda x: (sigmoid(x) > threshold).astype(int)
        case False:
            loss_func = F.cross_entropy
            pred_func = lambda x: np.argmax(x, axis=1)

    device = next(model.parameters()).device
    model = model.eval()
    test_loss = []
    true, pred = [], []
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y_hat = model(x)
        true.append(y.detach().cpu().numpy())
        pred.append(y_hat.detach().cpu().numpy())
        loss = loss_func(y_hat, y.to(device))
        test_loss.append(loss.item())
    mean_loss = sum(test_loss) / len(test_loss)
    print(f"Val loss:{mean_loss:0.4f}")
    true = np.concatenate(true)
    pred = pred_func(np.concatenate(pred))
    results = precision_recall_fscore_support(true, pred)
    print(f"{' ':<10} {'prc':<4} {'rec':<4} {'f1':<4} {'support'}")
    for l, (p, r, f1, s) in zip(dataloader.dataset.classes, zip(*results)):
        print(f"{l:<10} {p:0.2f} {r:0.2f} {f1:0.2f} {s: 6d}")
    macro_p, macro_r, macro_f1, _ = map(np.mean, results)
    print(f"{'mac. avg.':<10} {macro_p:0.2f} {macro_r:0.2f} {macro_f1:0.2f}")
    return mean_loss, macro_f1

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('label_schema', choices=['flat', 'hierarchical'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--gpu-id', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--model-dir', type=Path, default=Path('./models'))
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' 
                          if args.gpu_id is not None else 'cpu')
    torch.multiprocessing.set_sharing_strategy('file_system')

    match args.label_schema:

        case 'flat':
            make_dataset = CIFAR10Multilabel
            save_dir = args.model_dir/'vgg16_cifar10_flat'

        case 'hierarchical':
            make_dataset = CIFAR10Hierarchical
            save_dir = args.model_dir/'vgg16_cifar10_hierarchical'
   
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data = make_dataset(
        root=args.data_dir, 
        download=True,
        train=True, 
        transform=cifar10_transform
    )

    train_dataloader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=6
    )

    test_data = make_dataset(
        root=args.data_dir,  
        download=True,
        train=False, 
        transform=cifar10_transform
    )

    test_dataloader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=6
    )

    n_classes = len(train_data.classes)

    model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", 
            "cifar10_vgg16_bn", 
            pretrained=True
    )

    # re-initialize the final classifier layer
    model.classifier[-1] = nn.Linear(512, n_classes, bias=False)
    model = model.to(device)

    # optimizer = torch.optim.Adam(model.classifier[-1].parameters())
    optimizer = torch.optim.Adam(model.classifier.parameters())
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    train_log(save_dir, str(args), also_print=False)
    
    val_f1s = []
    for epoch in range(1, args.n_epochs+1):
        model = model.train()
        epoch_loss = []
        print(f"Epoch {epoch}.")
        for batch, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x = x.to(device)
            y_hat = model(x)

            loss = loss_fn(y_hat, y.to(device))

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        mean_loss = sum(epoch_loss) / len(epoch_loss)
        train_log(save_dir, f"Epoch {epoch} train loss:{mean_loss:0.4f}")

        torch.save(model.state_dict(), save_dir/'model.pt')
        val_loss, val_f1 = evaluate(
                test_dataloader, 
                model, 
                multilabel=True,
                threshold=0.5
        )
        train_log(save_dir, 
            f"Epoch {epoch} val loss:{val_loss:0.4f}; val f1:{val_f1:0.4f}"
        )

        if not val_f1s or val_f1 > max(val_f1s):
            train_log(save_dir, f"Saving E{epoch} model.")
            torch.save(model.state_dict(), save_dir/'model.pt')
        val_f1s.append(val_f1)


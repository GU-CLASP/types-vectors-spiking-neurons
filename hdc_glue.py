import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def evaluate_vsa_classifier(model, dataloader, train_totals=None, break_after=None):
    if train_totals is None:
        train_totals = [0] * len(dataloader.dataset.classes)
    true, pred = [], []
    for batch, (x, y) in enumerate(dataloader):
        x_dense = get_vgg_features(vgg, x)
        true.append(y.detach().cpu().numpy())
        x_dense = get_vgg_features(vgg, x)
        x_hd = hd_glue(x_dense)
        x_hd_M = torchhd.bind(M, x_hd)
        correlation = torch.stack([
            torchhd.bind(x_hd_M, class_ids[i]) 
                for i in range(n_classes)]
            ).swapaxes(0,1).sum(axis=-1)
        pred.append(correlation.argmax(dim=1))
        if break_after and batch >= break_after:
            break
    pred = np.concatenate(pred)
    true = np.concatenate(true)
    results = precision_recall_fscore_support(true, pred)
    print(f"{' ':<10} {'prc':<4} {'rec':<4} {'f1':<4} {'support'} {'n_train'}")
    for i, (l, (p, r, f1, s)) in enumerate(zip(dataloader.dataset.classes, zip(*results))):
        nt = train_totals[i]
        print(f"{l:<10} {p:0.2f} {r:0.2f} {f1:0.2f} {s: 7d} {nt: 7d}")
    macro_p, macro_r, macro_f1, _ = map(np.mean, results)
    print(f"{'mac. avg.':<10} {macro_p:0.2f} {macro_r:0.2f} {macro_f1:0.2f}")

    return macro_f1

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn as nn
import torch

cifar10_mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
cifar10_std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)

cifar10_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(cifar10_mean, cifar10_std)
])

def cifar10_dataloader(split, batch_size, shuffle=False):

    if not split in ('train', 'test'):
        raise ValueError(f"Invalid CIFAR-10 split {split}")

    dataset = CIFAR10(
        "./data", 
        train=split == 'train',
        download=True, 
        transform=cifar10_transform
    )

    return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=6
    )


def get_paths(tax):
    paths = {}
    def get_paths_(tax, prefix):
        if isinstance(tax, set):
            for item in tax:
                paths[item] = prefix + [item]
        elif isinstance(tax, dict):
            for k, v in tax.items():
                get_paths_(v, prefix + [k])
        else:
            raise ValueError
    get_paths_(tax, [])
    return paths

h_class_labels = ['cat', 'deer', 'dog', 'horse', 'bird', 'frog', 'automobile', 'truck', 'airplane', 'ship',
        'mammal', 'non-mammal', 'vehicle', 'craft', 'living', 'non-living', 'entity']
h_label_to_idx = {l:i for i,l in enumerate(h_class_labels)}

cifar10_hierarchy = {
    'entity': {
        'living': {
            'mammal': {'cat', 'deer', 'dog', 'horse'},
            'non-mammal': {'bird', 'frog'},
        },
        'non-living': {
            'vehicle': {'automobile', 'truck'},
            'craft': {'airplane', 'ship'}
        }
    }
}

class CIFAR10Hierarchical(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f_to_h = get_paths(cifar10_hierarchy)
        self.f_to_h_idxs = {
            self.class_to_idx[f]: list(map(h_label_to_idx.get, hs))
                for f,hs in f_to_h.items()
        }
        self.flat_classes = self.classes
        self.flat_class_to_idx = self.class_to_idx
        self.classes = h_class_labels
        self.class_to_idx = h_label_to_idx

    def __getitem__(self, index):
        img, flat_target = super().__getitem__(index)
        targets = torch.tensor(self.f_to_h_idxs[flat_target])
        targets = torch.zeros(len(self.classes)).scatter_(0, targets, 1.)
        return img, targets

def get_vgg_features(model, x):
    x = model.features(x)
    x = torch.flatten(x, 1)
    return model.classifier[:-3](x)

def retrain_for_multilabel(model, save_path):
    """
    Retrain a classification model for multi-label classification;
    i.e., using pointwise Sigmoid activation rather than Softmax
    and BCE loss rather than CEL loss.

    This removes the one-class-per-image/entity assumption. Inherent
    in the CIFAR-10 dataset (but not in e.g., CIFAR-100).
    """

    pass

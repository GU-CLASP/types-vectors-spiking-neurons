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

def get_tree_paths(tax):
    paths = {}
    def get_tree_paths_(tax, prefix):
        if isinstance(tax, set):
            for item in tax:
                paths[item] = prefix + [item]
        elif isinstance(tax, dict):
            for k, v in tax.items():
                get_tree_paths_(v, prefix + [k])
        else:
            raise ValueError
    get_tree_paths_(tax, [])
    return paths

flat_labels = [
        'cat', 
        'deer', 
        'dog', 
        'horse', 
        'bird', 
        'frog', 
        'automobile', 
        'truck', 
        'airplane', 
        'ship'
    ]

hierarchical_labels = flat_labels + [
        'mammal', 
        'non-mammal', 
        'vehicle', 
        'craft', 
        'living', 
        'non-living', 
        'entity'
    ]

flat_label_to_idx = {l:i for i,l in enumerate(flat_labels)}
hierarchical_label_to_idx = {l:i for i,l in enumerate(hierarchical_labels)}

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

class CIFAR10Multilabel(CIFAR10):
    """
    Thin wrpper around CIFAR10 so we return a list of labels instead of 
    a single label (the list is always just one item though).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        targets = torch.tensor([target])
        targets = torch.zeros(len(self.classes)).scatter_(0, targets, 1.)
        return img, targets


class CIFAR10Hierarchical(CIFAR10):
    """
    Wrapper around CIFAR10 that returns a list of labels given the "leaf label"
    of standard CIFAR10. Non-leaf labels are defined by `cifar10_hierachy`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flat_to_hierarchical = get_tree_paths(cifar10_hierarchy)
        self.flat_to_hierarchical_idxs = {
            self.class_to_idx[f]: list(map(hierarchical_label_to_idx.get, hs))
                for f,hs in flat_to_hierarchical.items()
        }
        self.flat_classes = self.classes
        self.flat_class_to_idx = self.class_to_idx
        self.classes = hierarchical_labels
        self.class_to_idx = hierarchical_label_to_idx

    def __getitem__(self, index):
        img, flat_target = super().__getitem__(index)
        targets = torch.tensor(self.flat_to_hierarchical_idxs[flat_target])
        targets = torch.zeros(len(self.classes)).scatter_(0, targets, 1.)
        return img, targets

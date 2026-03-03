import torch
import torchhd
import torch.nn.functional as F

from cifar10 import cifar10_dataloader

from itertools import islice

from pyttr.ttrtypes import Type, BType, Pred, PType, RecType, Fun
from pyttr.utils import show, show_latex, print_latex

from utils import show, show_latex
from records import Rec

def concensus_sum(x, dim=0):
    if False: #TODO Gemoved randomness for debuggin!!!! x.size(dim) % 2 == 0: # break ties randomly
        size = x.index_select(dim, torch.LongTensor([0])).squeeze().size()
        if len(size) == 1:
            r = torchhd.random(1,size[0], vsa='BSC').squeeze()
        elif len(size) == 2:
            r = torchhd.random(size[0],size[1], vsa='BSC')
        else:
            raise ValueError("too many dimensions")
        x = torch.cat((x, r.unsqueeze(dim)), dim=dim)
    threshold = x.size(dim) // 2
    return x.sum(dim) > threshold

def compact_concensus_sum(x, count):
    if count % 2 == 0:
        size = x.size(0)
        r = torchhd.random(1, size, vsa='BSC').squeeze()
        x += r
    threshold = count // 2
    return x > threshold

class HDGlue:

    def __init__(self, nn_dim, hyp_dim, n_bins):

        boundary_map = torch.vmap(lambda x: x/((n_bins-1)/2)-1)
        self.tanh_bounds = boundary_map(torch.LongTensor(range(n_bins)))
        self.component_ids = torchhd.random(nn_dim, hyp_dim, vsa='BSC')
        self.bin_reps = torch.stack([
            torchhd.level(n_bins, hyp_dim, vsa='BSC') 
                for _ in range(nn_dim)
        ])

    _embedding = staticmethod(torch.vmap(F.embedding, in_dims=0))

    def __call__(self, x):
        batch_size = x.size(0)
        # pointwise tanh features to [-1, 1]
        x = F.tanh(x)
        # compute bin for each tanh value
        x = torch.bucketize(x, self.tanh_bounds)
        # embed values using HD bin representations
        x = self._embedding(x.T, self.bin_reps).transpose(0,1)
        # bind component values to component IDs
        x = self.component_ids.repeat([batch_size, 1, 1]).bind(x)
        # concensus sum along the component dimension
        x = concensus_sum(x, dim=1)
        return x


def train_model(dataloader, vision_model, class_ids):
    
    memory_totals = torch.zeros(n_classes, dtype=torch.long)
    memory = torch.zeros((n_classes, hyp_dim), dtype=torch.long)

    for i, (x, y) in enumerate(train_dataloader):

        x_feats = vision_model.get_features(x)
        x_hd = hd_glue(x_feats)
        memory_totals += torch.bincount(y, minlength=n_classes)
        for i in range(n_classes):
            # mask out other classes
            x_i = x_hd * (y == i).unsqueeze(1) 
            # add component counts to memory
            memory[i] += x_i.sum(dim = 0)

        if (i * batch_size) % 100 == 0:
            break

    prototypes = torch.stack(
        [compact_concensus_sum(memory[i], memory_totals[i])
             for i in range(n_classes)], dim=1
    ).T

    # TODO: online traning                            
    return concensus_sum(torchhd.bind(class_ids, prototypes))


n_bins = 100
hyp_dim = 2**12
vis_dim = 512
batch_size = 64
n_classes = 10

vgg = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", 
    "cifar10_vgg16_bn", 
    pretrained=True
).eval()

hd_glue = HDGlue(vis_dim, hyp_dim, n_bins)

test_dataloader = cifar10_dataloader('test', batch_size)
train_dataloader = cifar10_dataloader('train', batch_size)

class_ids = torchhd.random(n_classes, hyp_dim, vsa='BSC')
model = train_model(islice(train_dataloader, None, 5), vgg, class_ids)

from IPython import embed; embed(colors="neutral"); raise;

for (x, y) in islice(test_dataloader, None, 5):
    x_hd = hd_glue(vgg.get_features(x))
    break


for i in range(n_classes):
    print((torchhd.bind(model, class_ids[i]) == x_hd[0]).sum().item())

from IPython import embed; embed(); raise;

Ind = BType('Ind')
hug = Pred('hug',[Ind,Ind])

boy = Pred('boy',[Ind])
dog = Pred('dog',[Ind])

a_boy_hugs_a_dog = RecType({'x':Ind,
                            'c_boy':(Fun('v',Ind,PType(boy,['v'])), ['x']),
                            'y':Ind,
                            'c_dog':(Fun('v',Ind,PType(dog,['v'])), ['y']),
                            'e':(Fun('v1',Ind,Fun('v2',Ind, PType(hug,['v1','v2']))), 
                                     ['x','y'])})

print(show(a_boy_hugs_a_dog))
show_latex(a_boy_hugs_a_dog)



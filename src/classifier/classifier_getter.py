from dataset.utils import tprint
from model.r2d2 import R2D2
from model.mlp import MLP


def get_classifier(ebd_dim, args):
    tprint("Building classifier")

    model = MLP(ebd_dim, args)

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
    

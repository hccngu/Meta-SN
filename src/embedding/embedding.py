import datetime
from embedding.wordebd import WORDEBD

from model.simaese_model import ModelG


def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    ebd = WORDEBD(vocab, args.finetune_ebd)

    modelG = ModelG(ebd, args)
    # modelD = ModelD(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.cuda != -1:
        modelG = modelG.cuda(args.cuda)
        # modelD = modelD.cuda(args.cuda)
        return modelG  # , modelD
    else:
        return modelG  # , modelD
    

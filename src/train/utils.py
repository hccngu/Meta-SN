import itertools
import torch

import torch.nn.functional as F



def named_grad_param(model, keys):
    '''
        Return a generator that generates learnable named parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p[1].requires_grad,
                model[keys[0]].named_parameters())
    else:
        return filter(lambda p: p[1].requires_grad,
                itertools.chain.from_iterable(
                    model[key].named_parameters() for key in keys))


def grad_param(model, keys):
    '''
        Return a generator that generates learnable parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p.requires_grad,
                model[keys[0]].parameters())
    else:
        return filter(lambda p: p.requires_grad,
                itertools.chain.from_iterable(
                    model[key].parameters() for key in keys))


def get_norm(model):
    '''
        Compute norm of the gradients
    '''
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            p_norm = p.grad.data.norm()
            total_norm += p_norm.item() ** 2

    total_norm = total_norm ** 0.5

    return total_norm




def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()



def neg_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def pos_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def dis_to_level(dis):
    tmp_mean = torch.mean(dis, dim=-1, keepdim=True)
    result = dis / tmp_mean
    return -result

# 自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label, weight):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        euclidean_distance = euclidean_distance / torch.mean(euclidean_distance)

        tmp1 = (label) * torch.pow(euclidean_distance, 2).squeeze(-1)
        # mean_val = torch.mean(euclidean_distance)
        tmp2 = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                       2).squeeze(-1)
        loss_contrastive = torch.mean((tmp1 + tmp2)*weight)

        # print("**********************************************************************")
        return loss_contrastive



def get_weight_of_test_support(support, query, args):
    if len(support) > args.way*args.shot:
        support = support[0:args.way*args.shot]
    result = torch.cat( (torch.ones([args.way*args.shot*args.way]), args.test_loss_weight*torch.ones([args.way*args.way])),0 )

    tensor_shape = support.shape[-1]
    for each_way in range(args.way):
        this_support = support[each_way*args.shot:each_way*args.shot+args.shot]
        this_query = query[each_way*args.query:each_way*args.query+args.query]
        all_dis = torch.ones([args.shot])
        new_support = torch.ones([args.query,tensor_shape])
        for each_shot in range(args.shot):
            new_support[:] = this_support[each_shot]
            new_support = new_support.cuda(args.cuda)
            this_dis = F.pairwise_distance(new_support, this_query, keepdim=True)
            this_dis = torch.mean(this_dis)
            all_dis[each_shot] = this_dis
        probab = dis_to_level(all_dis)
        probab = F.softmax(probab, dim = -1)
        probab = 5*probab
        for each_shot in range(args.shot):
            begin = each_way*(args.shot*args.way)+each_shot*args.way
            result[begin:begin+args.way] = probab[each_shot]

    if args.cuda != -1:
        result = result.cuda(args.cuda)
    return result


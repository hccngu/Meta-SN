import datetime


import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
import copy

from train.utils import *
from dataset.sampler2 import SerialSampler,  task_sampler
from dataset import utils
from tools.tool import reidx_y, neg_dist

from torch import autograd
from collections import OrderedDict


def test_one(task, class_names, model, optG, criterion, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['G'].eval()

    support, query = task
    # print("support, query:", support, query)
    # print("class_names_dict:", class_names_dict)

    '''分样本对'''
    YS = support['label']
    YQ = query['label']

    sampled_classes = torch.unique(support['label']).cpu().numpy().tolist()
    # print("sampled_classes:", sampled_classes)

    class_names_dict = {}
    class_names_dict['label'] = class_names['label'][sampled_classes]
    # print("class_names_dict['label']:", class_names_dict['label'])
    class_names_dict['text'] = class_names['text'][sampled_classes]
    class_names_dict['text_len'] = class_names['text_len'][sampled_classes]
    class_names_dict['is_support'] = False
    class_names_dict = utils.to_tensor(class_names_dict, args.cuda, exclude_keys=['is_support'])

    YS, YQ = reidx_y(args, YS, YQ)
    # print('YS:', support['label'])
    # print('YQ:', query['label'])
    # print("class_names_dict:", class_names_dict['label'])

    """维度填充"""
    if support['text'].shape[1] > class_names_dict['text'].shape[1]:
        zero = torch.zeros(
            (class_names_dict['text'].shape[0], support['text'].shape[1] - class_names_dict['text'].shape[1]),
            dtype=torch.long)
        class_names_dict['text'] = torch.cat((class_names_dict['text'], zero.cuda()), dim=-1)
    elif support['text'].shape[1] < class_names_dict['text'].shape[1]:
        zero = torch.zeros(
            (support['text'].shape[0], class_names_dict['text'].shape[1] - support['text'].shape[1]),
            dtype=torch.long)
        support['text'] = torch.cat((support['text'], zero.cuda()), dim=-1)

    support['text'] = torch.cat((support['text'], class_names_dict['text']), dim=0)
    support['text_len'] = torch.cat((support['text_len'], class_names_dict['text_len']), dim=0)
    support['label'] = torch.cat((support['label'], class_names_dict['label']), dim=0)
    # print("support['text']:", support['text'].shape)
    # print("support['label']:", support['label'])

    text_sample_len = support['text'].shape[0]
    # print("support['text'].shape[0]:", support['text'].shape[0])
    support['text_1'] = support['text'][0].view((1, -1))
    support['text_len_1'] = support['text_len'][0].view(-1)
    support['label_1'] = support['label'][0].view(-1)
    for i in range(text_sample_len):
        if i == 0:
            for j in range(1, len(sampled_classes)):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)
        else:
            for j in range(len(sampled_classes)):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)

    support['text_2'] = class_names_dict['text'][0].view((1, -1))
    support['text_len_2'] = class_names_dict['text_len'][0].view(-1)
    support['label_2'] = class_names_dict['label'][0].view(-1)
    for i in range(text_sample_len):
        if i == 0:
            for j in range(1, len(sampled_classes)):
                support['text_2'] = torch.cat((support['text_2'], class_names_dict['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)),dim=0)
                support['label_2'] = torch.cat((support['label_2'], class_names_dict['label'][j].view(-1)), dim=0)
        else:
            for j in range(len(sampled_classes)):
                support['text_2'] = torch.cat((support['text_2'], class_names_dict['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)),dim=0)
                support['label_2'] = torch.cat((support['label_2'], class_names_dict['label'][j].view(-1)), dim=0)

    # print("support['text_1']:", support['text_1'].shape, support['text_len_1'].shape, support['label_1'].shape)
    # print("support['text_2']:", support['text_2'].shape, support['text_len_2'].shape, support['label_2'].shape)
    support['label_final'] = support['label_1'].eq(support['label_2']).int()

    support_1 = {}
    support_1['text'] = support['text_1']
    support_1['text_len'] = support['text_len_1']
    support_1['label'] = support['label_1']

    support_2 = {}
    support_2['text'] = support['text_2']
    support_2['text_len'] = support['text_len_2']
    support_2['label'] = support['label_2']
    # print("**************************************")
    # print("1111111", support['label_1'])
    # print("2222222", support['label_2'])
    # print(support['label_final'])

    '''first step'''
    S_out1, S_out2 = model['G'](support_1, support_2)

    supp_, que_ = model['G'](support, query)
    loss_weight = get_weight_of_test_support(supp_, que_, args)

    loss = criterion(S_out1, S_out2, support['label_final'], loss_weight)
    # print("s_1_loss:", loss)
    zero_grad(model['G'].parameters())

    grads_fc = autograd.grad(loss, model['G'].fc.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_fc, orderd_params_fc = model['G'].cloned_fc_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].fc.named_parameters(), grads_fc):
        fast_weights_fc[key] = orderd_params_fc[key] = val - args.task_lr * grad

    grads_conv11 = autograd.grad(loss, model['G'].conv11.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv11, orderd_params_conv11 = model['G'].cloned_conv11_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv11.named_parameters(), grads_conv11):
        fast_weights_conv11[key] = orderd_params_conv11[key] = val - args.task_lr * grad

    grads_conv12 = autograd.grad(loss, model['G'].conv12.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv12, orderd_params_conv12 = model['G'].cloned_conv12_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv12.named_parameters(), grads_conv12):
        fast_weights_conv12[key] = orderd_params_conv12[key] = val - args.task_lr * grad

    grads_conv13 = autograd.grad(loss, model['G'].conv13.parameters(), allow_unused=True)
    fast_weights_conv13, orderd_params_conv13 = model['G'].cloned_conv13_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv13.named_parameters(), grads_conv13):
        fast_weights_conv13[key] = orderd_params_conv13[key] = val - args.task_lr * grad


    fast_weights = {}
    fast_weights['fc'] = fast_weights_fc
    fast_weights['conv11'] = fast_weights_conv11
    fast_weights['conv12'] = fast_weights_conv12
    fast_weights['conv13'] = fast_weights_conv13

    '''steps remaining'''
    for k in range(args.test_iter - 1):
        S_out1, S_out2 = model['G'](support_1, support_2, fast_weights)

        supp_, que_ = model['G'](support, query, fast_weights)
        loss_weight = get_weight_of_test_support(supp_, que_, args)

        loss = criterion(S_out1, S_out2, support['label_final'], loss_weight)
        # print("train_iter: {} s_loss:{}".format(k, loss))
        zero_grad(orderd_params_fc.values())
        zero_grad(orderd_params_conv11.values())
        zero_grad(orderd_params_conv12.values())
        zero_grad(orderd_params_conv13.values())
        grads_fc = torch.autograd.grad(loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
        grads_conv11 = torch.autograd.grad(loss, orderd_params_conv11.values(), allow_unused=True, retain_graph=True)
        grads_conv12 = torch.autograd.grad(loss, orderd_params_conv12.values(), allow_unused=True, retain_graph=True)
        grads_conv13 = torch.autograd.grad(loss, orderd_params_conv13.values(), allow_unused=True)

        for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
            if grad is not None:
                fast_weights['fc'][key] = orderd_params_fc[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv11.items(), grads_conv11):
            if grad is not None:
                fast_weights['conv11'][key] = orderd_params_conv11[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv12.items(), grads_conv12):
            if grad is not None:
                fast_weights['conv12'][key] = orderd_params_conv12[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv13.items(), grads_conv13):
            if grad is not None:
                fast_weights['conv13'][key] = orderd_params_conv13[key] = val - args.task_lr * grad

    """计算Q上的损失"""
    CN = model['G'].forward_once_with_param(class_names_dict, fast_weights)
    XQ = model['G'].forward_once_with_param(query, fast_weights)
    logits_q = pos_dist(XQ, CN)
    logits_q = dis_to_level(logits_q)
    _, pred = torch.max(logits_q, 1)
    acc_q = model['G'].accuracy(pred, YQ)

    return acc_q


def test(test_data, class_names, optG, model, criterion, args, test_epoch, verbose=True):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    # model['G'].train()

    acc = []

    for ep in range(test_epoch):
        #print("**********************test ep******************************",ep)

        sampled_classes, source_classes = task_sampler(test_data, args)

        train_gen = SerialSampler(test_data, args, sampled_classes, source_classes, 1)

        sampled_tasks = train_gen.get_epoch()

        for task in sampled_tasks:
            if task is None:
                break
            q_acc = test_one(task, class_names, model, optG, criterion, args, grad={})
            acc.append(q_acc.cpu().item())

    acc = np.array(acc)

    if verbose:
        if args.embedding != 'mlada':
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now(),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
            ), flush=True)
        else:
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now(),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
            ), flush=True)

    return np.mean(acc), np.std(acc)

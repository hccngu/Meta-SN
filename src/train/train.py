import os
import time
import datetime

import torch
import numpy as np
import copy

from train.utils import *
from dataset.sampler2 import SerialSampler,  task_sampler
from tqdm import tqdm
from termcolor import colored
from train.test import test
import torch.nn.functional as F
from dataset import utils
from tools.tool import neg_dist, reidx_y

from torch import autograd
from collections import OrderedDict




def del_tensor_ele(arr, index):
        arr1 = arr[0:index]
        arr2 = arr[index + 1:]
        return torch.cat((arr1, arr2), dim=0)


def pre_calculate(train_data, class_names, net, args):
    with torch.no_grad():
        all_classes = np.unique(train_data['label'])
        num_classes = len(all_classes)

        # 生成sample类时候的概率矩阵
        train_class_names = {}
        train_class_names['text'] = class_names['text'][all_classes]
        train_class_names['text_len'] = class_names['text_len'][all_classes]
        train_class_names['label'] = class_names['label'][all_classes]
        train_class_names = utils.to_tensor(train_class_names, args.cuda)
        train_class_names_ebd = net.ebd(train_class_names)  # [10, 36, 300]
        train_class_names_ebd = torch.sum(train_class_names_ebd, dim=1) / train_class_names['text_len'].view((-1, 1))  # [10, 300]
        dist_metrix = -neg_dist(train_class_names_ebd, train_class_names_ebd)  # [10, 10]

        for i, d in enumerate(dist_metrix):
            if i == 0:
                dist_metrix_nodiag = del_tensor_ele(d, i).view((1, -1))
            else:
                dist_metrix_nodiag = torch.cat((dist_metrix_nodiag, del_tensor_ele(d, i).view((1, -1))), dim=0)

        prob_metrix = F.softmax(dist_metrix_nodiag, dim=1)  # [10, 9]
        prob_metrix = prob_metrix.cpu().numpy()


        # 生成sample样本时候的概率矩阵
        example_prob_metrix = []
        for i, label in enumerate(all_classes):
            train_examples = {}
            train_examples['text'] = train_data['text'][train_data['label'] == label]
            train_examples['text_len'] = train_data['text_len'][train_data['label'] == label]
            train_examples['label'] = train_data['label'][train_data['label'] == label]
            train_examples = utils.to_tensor(train_examples, args.cuda)
            train_examples_ebd = net.ebd(train_examples)
            train_examples_ebd = torch.sum(train_examples_ebd, dim=1) / train_examples['text_len'].view(
                                    (-1, 1))  # [N, 300]
            example_prob_metrix_one = -neg_dist(train_class_names_ebd[i].view((1, -1)), train_examples_ebd)
            example_prob_metrix_one = F.softmax(example_prob_metrix_one, dim=1)  # [1, 1000]
            example_prob_metrix_one = example_prob_metrix_one.cpu().numpy()
            example_prob_metrix.append(example_prob_metrix_one)

        return prob_metrix, example_prob_metrix


def train_one(task, class_names, model, optG, criterion, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['G'].train()
    # model['G2'].train()
    # model['clf'].train()

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


    support['label_final'] = support['label_1'].eq(support['label_2']).int()

    support_1 = {}
    support_1['text'] = support['text_1']
    support_1['text_len'] = support['text_len_1']
    support_1['label'] = support['label_1']

    support_2 = {}
    support_2['text'] = support['text_2']
    support_2['text_len'] = support['text_len_2']
    support_2['label'] = support['label_2']



    '''first step'''
    S_out1, S_out2 = model['G'](support_1, support_2)
    # print("-------0S1_2:", S_out1.shape, S_out2.shape)

    # supp_, que_ = model['G'](support, query)
    # loss_weight = get_weight_of_support(supp_, que_, args)

    loss_weight = torch.cat(
        (torch.ones([args.way * args.shot * args.way]), args.train_loss_weight * torch.ones([args.way * args.way])), 0)
    if args.cuda != -1:
        loss_weight = loss_weight.cuda(args.cuda)

    loss = criterion(S_out1, S_out2, support['label_final'], loss_weight)
    # print("**********loss first step*******", loss)
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
    for k in range(args.train_iter - 1):
        S_out1, S_out2 = model['G'](support_1, support_2, fast_weights)
        # print("-------1S1_2:", S_out1, S_out2)
        # supp_, que_ = model['G'](support, query, fast_weights)
        # loss_weight = get_weight_of_support(supp_, que_, args)

        loss_weight = torch.cat(
            (torch.ones([args.way * args.shot * args.way]), args.train_loss_weight * torch.ones([args.way * args.way])),
            0)
        if args.cuda != -1:
            loss_weight = loss_weight.cuda(args.cuda)

        loss = criterion(S_out1, S_out2, support['label_final'], loss_weight)
        # print("**********loss remain step*******", loss)
        # print("train_iter: {} s_loss:{}".format(k, loss))
        zero_grad(orderd_params_fc.values())
        zero_grad(orderd_params_conv11.values())
        zero_grad(orderd_params_conv12.values())
        zero_grad(orderd_params_conv13.values())
        grads_fc = torch.autograd.grad(loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
        grads_conv11 = torch.autograd.grad(loss, orderd_params_conv11.values(), allow_unused=True, retain_graph=True)
        grads_conv12 = torch.autograd.grad(loss, orderd_params_conv12.values(), allow_unused=True, retain_graph=True)
        grads_conv13 = torch.autograd.grad(loss, orderd_params_conv13.values(), allow_unused=True)
        # print('grads:', grads)
        # print("orderd_params.items():", orderd_params.items())
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
    # print("logits_q:", logits_q)
    logits_q = dis_to_level(logits_q)
    q_loss = model['G'].loss(logits_q, YQ)
    # print("q_loss:", q_loss)
    _, pred = torch.max(logits_q, 1)
    acc_q = model['G'].accuracy(pred, YQ)

    return q_loss, acc_q


def train(train_data, val_data, model, class_names, criterion, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
        os.path.curdir,
        "tmp-runs",
        str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    if args.STS == True:
        classes_sample_p, example_prob_metrix = pre_calculate(train_data, class_names, model['G'], args)
    else:
        classes_sample_p, example_prob_metrix = None, None

    optG = torch.optim.Adam(grad_param(model, ['G']), lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optG, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    elif args.lr_scheduler == 'ExponentialLR':
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=args.ExponentialLR_gamma)

    print("{}, Start training".format(
        datetime.datetime.now()), flush=True)

    acc = 0
    loss = 0
    for ep in range(args.train_epochs):
        ep_loss = 0
        for _ in range(args.train_episodes):

            sampled_classes, source_classes = task_sampler(train_data, args, classes_sample_p)

            train_gen = SerialSampler(train_data, args, sampled_classes, source_classes, 1, example_prob_metrix)

            sampled_tasks = train_gen.get_epoch()

            grad = {'clf': [], 'G': []}

            if not args.notqdm:
                sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                                     ncols=80, leave=False, desc=colored('Training on train',
                                                                         'yellow'))

            for task in sampled_tasks:
                if task is None:
                    break
                q_loss, q_acc = train_one(task, class_names, model, optG, criterion, args, grad)
                acc += q_acc
                loss = loss + q_loss
                ep_loss = ep_loss + q_loss

        ep_loss = ep_loss / args.train_episodes

        optG.zero_grad()
        ep_loss.backward()
        optG.step()

        test_count = 100
        if (ep % test_count == 0) and (ep != 0):
            acc = acc / args.train_episodes / test_count
            loss = loss / args.train_episodes / test_count
            print("{}:".format(colored('--------[TRAIN] ep', 'blue')) + str(ep) + ", mean_loss:" + str(loss.item()) + ", mean_acc:" + str(
                acc.item()) + "-----------")

            net = copy.deepcopy(model)

            acc = 0
            loss = 0

            # Evaluate validation accuracy
            cur_acc, cur_std = test(val_data, class_names, optG, net, criterion, args, args.val_epochs, False)
            print(("[EVAL] {}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
                   ).format(
                datetime.datetime.now(),
                "ep", ep,
                colored("val  ", "cyan"),
                colored("acc:", "blue"), cur_acc, cur_std,
            ), flush=True)

            # Update the current best model if val acc is better
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_path = os.path.join(out_dir, str(ep))

                # save current model
                print("{}, Save cur best model to {}".format(
                    datetime.datetime.now(),
                    best_path))
                torch.save(model['G'].state_dict(), best_path + '.G')

                sub_cycle = 0
            else:
                sub_cycle += 1

            # Break if the val acc hasn't improved in the past patience epochs
            if sub_cycle == args.patience:
                break

            if args.lr_scheduler == 'ReduceLROnPlateau':
                schedulerG.step(cur_acc)
                # schedulerCLF.step(cur_acc)

            elif args.lr_scheduler == 'ExponentialLR':
                schedulerG.step()
                # schedulerCLF.step()

    print("{}, End of training. Restore the best weights".format(
        datetime.datetime.now()),
        flush=True)

    # restore the best saved model
    model['G'].load_state_dict(torch.load(best_path + '.G'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir,
            "saved-runs",
            str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now(),
            best_path), flush=True)

        torch.save(model['G'].state_dict(), best_path + '.G')
        # torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return optG


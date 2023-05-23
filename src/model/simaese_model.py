import torch
import torch.nn as nn

import torch.nn.functional as F



class ModelG(nn.Module):

    def __init__(self, ebd, args):
        super(ModelG, self).__init__()

        self.args = args

        self.ebd = ebd
        self.ebd_begin_len = args.ebd_len

        self.ebd_dim = self.ebd.embedding_dim
        self.hidden_size = 128

        # Text CNN
        ci = 1  # input chanel size
        kernel_num = args.kernel_num  # output chanel size
        kernel_size = args.kernel_size
        dropout = args.dropout
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], self.ebd_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], self.ebd_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], self.ebd_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_size) * kernel_num, 64)
        self.cost = nn.CrossEntropyLoss()

    def forward_once(self, data):

        ebd = self.ebd(data)  # [b, text_len, 300]
        # if data['text_len'][0] < 60:
        #     ebd = ebd[:, :self.ebd_begin_len, :]
        ebd = ebd[:, :self.ebd_begin_len, :]
        ebd = ebd.unsqueeze(1)  # [b, 1, text_len, 300]

        x1 = self.conv11(ebd)  # [b, kernel_num, H_out, 1]
        # print("conv11", x1.shape)
        x1 = F.relu(x1.squeeze(3))  # [b, kernel_num, H_out]
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch, kernel_num]

        x2 = self.conv12(ebd)  # [b, kernel_num, H_out, 1]
        # print("conv11", x2.shape)
        x2 = F.relu(x2.squeeze(3))  # [b, kernel_num, H_out]
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # [batch, kernel_num]

        x3 = self.conv13(ebd)  # [b, kernel_num, H_out, 1]
        # print("conv11", x3.shape)
        x3 = F.relu(x3.squeeze(3))  # [b, kernel_num, H_out]
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)  # [b, kernel_num]

        x = torch.cat((x1, x2, x3), 1)  # [b, 3 * kernel_num]
        # x = self.dropout(x)

        x = self.fc(x)  # [b, 128]
        x = self.dropout(x)

        return x

    def forward_once_with_param(self, data, param):

        ebd = self.ebd(data)  # [b, text_len, 300]
        # if data['text_len'][0] < 60:
        #     ebd = ebd[:, :self.ebd_begin_len, :]
        ebd = ebd.unsqueeze(1)  # [b, 1, text_len, 300]

        w1, b1 = param['conv11']['weight'], param['conv11']['bias']
        x1 = F.conv2d(ebd, w1, b1)  # [b, kernel_num, H_out, 1]
        # print("conv11", x1.shape)
        x1 = F.relu(x1.squeeze(3))  # [b, kernel_num, H_out]
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch, kernel_num]

        w2, b2 = param['conv12']['weight'], param['conv12']['bias']
        x2 = F.conv2d(ebd, w2, b2)  # [b, kernel_num, H_out, 1]
        # print("conv11", x2.shape)
        x2 = F.relu(x2.squeeze(3))  # [b, kernel_num, H_out]
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # [batch, kernel_num]

        w3, b3 = param['conv13']['weight'], param['conv13']['bias']
        x3 = F.conv2d(ebd, w3, b3)  # [b, kernel_num, H_out, 1]
        # print("conv11", x3.shape)
        x3 = F.relu(x3.squeeze(3))  # [b, kernel_num, H_out]
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)  # [b, kernel_num]

        x = torch.cat((x1, x2, x3), 1)  # [b, 3 * kernel_num]
        # x = self.dropout(x)

        w_fc, b_fc = param['fc']['weight'], param['fc']['bias']
        x = F.linear(x, w_fc, b_fc)  # [b, 128]
        x = self.dropout(x)

        return x

    def forward(self, inputs_1, inputs_2, param=None):
        if param is None:
            out_1 = self.forward_once(inputs_1)
            out_2 = self.forward_once(inputs_2)
        else:
            out_1 = self.forward_once_with_param(inputs_1, param)
            out_2 = self.forward_once_with_param(inputs_2, param)
        return out_1, out_2

    def cloned_fc_dict(self):
        return {key: val.clone() for key, val in self.fc.state_dict().items()}

    def cloned_conv11_dict(self):
        return {key: val.clone() for key, val in self.conv11.state_dict().items()}

    def cloned_conv12_dict(self):
        return {key: val.clone() for key, val in self.conv12.state_dict().items()}

    def cloned_conv13_dict(self):
        return {key: val.clone() for key, val in self.conv13.state_dict().items()}

    def loss(self, logits, label):
        loss_ce = self.cost(logits, label)
        return loss_ce

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label).type(torch.FloatTensor))


    

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

def mixup_data(x, y, runs, alpha=0.1, use_cuda=True):
    for i in range(runs):
        output_x = torch.Tensor(0)
        output_x= output_x.numpy().tolist()
        output_y = torch.Tensor(0)
        output_y = output_y.numpy().tolist()
        batch_size = x.size()[0]
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        output_x.append(mixed_x)
        output_y.append(mixed_y)
    return torch.cat(output_x,dim=0), torch.cat(output_y,dim=0)


def mixup_data_refactor( x, y, x_refactor, y_refactor, alpha, runs, use_cuda=True):
    for i in range(runs):
        output_x = torch.Tensor(0)
        output_x= output_x.numpy().tolist()
        output_y = torch.Tensor(0)
        output_y = output_y.numpy().tolist()
        batch_size = x.size()[0]
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x_refactor[index, :]
        mixed_y = lam * y + (1 - lam) * y_refactor[index, :]
        output_x.append(mixed_x)
        output_y.append(mixed_y)
    return torch.cat(output_x,dim=0), torch.cat(output_y,dim=0)


class Model(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.linear = nn.Linear(768, args.num_labels)
        

    def forward(self, input_ids=None,labels=None):
        
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        # [2, 256, 768]
        print("2:", outputs.shape)
        outputs = (outputs*input_ids.ne(1)[:,:,None]).sum(1)/input_ids.ne(1).sum(-1)[:,None]
        # [2, 768]
        output_x = torch.nn.functional.normalize(outputs, p=2, dim=1)
        print("3:", output_x.shape)
        output_xx = self.linear(output_x)
        print("4:", output_xx.shape)
        prob=torch.nn.functional.log_softmax(output_xx,-1)
        print("5:", prob.shape)

        if labels is not None:
            loss = -torch.sum(prob*labels)
            print(loss, prob.shape)
            return loss,prob
        else:
            return prob
        
        '''
        print("input_ids:{}".format(input_ids.shape))
        # logits=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        print("logits:{}".format(logits.shape))
        print("labels:{}".format(labels.shape))
        logits, labels = mixup_data(logits,labels,10) # Mixup Data 
        prob=torch.nn.functional.log_softmax(logits,-1)
        if labels is not None:
            loss = -torch.sum(prob*labels)
            return loss,prob
        else:
            return prob

        '''

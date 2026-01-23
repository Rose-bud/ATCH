import argparse
from logging import getLogger
import pickle
import os

import numpy as np
import torch

# from .logger import create_logger, PD_Stats

import torch.distributed as dist
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import network


def final_Confidence_Upadate(outputs1, outputs2, all_outputs1, all_outputs2, partialY, confidence, ind):
    with torch.no_grad():
        outputs_norm1 = torch.nn.functional.normalize(outputs1, dim=1)
        outputs_norm2 = torch.nn.functional.normalize(outputs2, dim=1)
        all_outputs_norm1 = torch.nn.functional.normalize(all_outputs1, dim=1)
        all_outputs_norm2 = torch.nn.functional.normalize(all_outputs2, dim=1)
    
        all_sim1 = torch.mm(all_outputs_norm1, all_outputs_norm1.t())
        all_sim1.fill_diagonal_(0)  
        all_sim2 = torch.mm(all_outputs_norm2, all_outputs_norm2.t())
        all_sim2.fill_diagonal_(0)
        all_sim = (all_sim1+all_sim2)/2
        top_sim, nearest_indices = torch.topk(all_sim, k=30, dim=1)
        
        batch_top_sim = top_sim[ind]
        batch_nearest_indices = nearest_indices[ind]

        batch_top_sim_norm = batch_top_sim/ torch.norm(batch_top_sim, dim=1, keepdim=True)

        neighbor_Y = partialY[batch_nearest_indices]
        neighbor_Y = neighbor_Y*(batch_top_sim_norm.unsqueeze(-1))
        neighbor_pesudo_Y = torch.sum(neighbor_Y, dim=1)
        neighbor_pesudo_Y = neighbor_pesudo_Y*partialY[ind]

        neighbor_pesudo_confidence = neighbor_pesudo_Y / (torch.sum(neighbor_pesudo_Y, dim=1).reshape(-1, 1)+1e-8)

    return neighbor_pesudo_confidence



def max_second_max_diff_scaled(x: torch.Tensor) -> torch.Tensor:
    max_values, big_index = torch.max(x, dim=1)
    
    top2_values, two_index = torch.topk(x, k=2, dim=1)
    first_index = two_index[:,0][0]
    second_index = two_index[:,1][0]
    
    second_max_values = top2_values[:, 1]
    
    differences = max_values - second_max_values
    
    min_diff = torch.min(differences)
    max_diff = torch.max(differences)
    scaled_differences = (differences - min_diff) / (max_diff - min_diff + 1e-8) 
    
    return scaled_differences

def LCD_loss(u_partial, v_partial, partialY, confidence, ind):
    
    confidence_batch = confidence[ind]
    weights1 = max_second_max_diff_scaled(confidence_batch)
   
    logsm_outputs_u = F.log_softmax(u_partial, dim=1)
    final_outputs_u = logsm_outputs_u * confidence[ind]
    average_loss_u = - ((final_outputs_u).sum(dim=1))

    logsm_outputs_v = F.log_softmax(v_partial, dim=1)
    final_outputs_v = logsm_outputs_v * confidence[ind]
    average_loss_v = - ((final_outputs_v).sum(dim=1))

    average_loss =  (average_loss_u + average_loss_v)/2

    return average_loss  , weights1


###################################################
##################CACH##################
###################################################
class CACH_loss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CACH_loss, self).__init__()
        self.shift = 1
        self.margin = 0.2
        self.tau = 1.0

    def forward(self, u, v, y, confidence, config):
        label1 = torch.argmax(confidence, dim=1)
        py = F.one_hot(label1).float()
        T = self.calc_neighbor(py, py)

        T_Total = T.numel()
        T_one = T.sum()
        T_zero = T_Total - T_one
        S = u.mm(v.t())
        d = S.diag().view(v.size(0), 1)
        d1 = d.expand_as(S)
        d2 = d.t().expand_as(S)

        mask_te = (S >= (d1 - self.margin)).float().detach()  
        cost_te = S * mask_te + (1.0 - mask_te) * (S - self.shift)

        cost_te_max = torch.zeros_like(cost_te)
        cost_te_max.copy_(cost_te)
        identity_matrix_te = torch.eye(
            cost_te_max.size(0),
            cost_te_max.size(1),
            device=cost_te_max.device,
            dtype=cost_te_max.dtype,
        )
        diagonal_te = torch.diag(cost_te_max).clamp(min=0)
        modified_diagonal_matrix_te = torch.diag_embed(diagonal_te)
        cost_te_max = (
            cost_te_max * (1 - identity_matrix_te) + modified_diagonal_matrix_te
        )

        mask_im = (S >= (d2 - self.margin)).float().detach()
        cost_im = S * mask_im + (1.0 - mask_im) * (S - self.shift)

        cost_im_max = torch.zeros_like(cost_im)
        cost_im_max.copy_(cost_im)
        identity_matrix_im = torch.eye(
            cost_im_max.size(0),
            cost_im_max.size(1),
            device=cost_im_max.device,
            dtype=cost_im_max.dtype,
        )
        diagonal_im = torch.diag(cost_im_max).clamp(min=0)
        modified_diagonal_matrix_im = torch.diag_embed(diagonal_im)
        cost_im_max = (
            cost_im_max * (1 - identity_matrix_im) + modified_diagonal_matrix_im
        )

        loss_r = (
            -cost_te.diag()
            + self.tau * ((cost_te_max / self.tau * (1 - T))).exp().sum(1).log()
            + self.margin
        ) + (
            -cost_im.diag()
            + self.tau * ((cost_im_max / self.tau * (1 - T))).exp().sum(1).log()
            + self.margin
        )
        
        return loss_r

    def calc_neighbor(self, label1, label2):
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
        return Sim
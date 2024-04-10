
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from torch.nn.init import xavier_uniform_
import copy
import random
from dis_causal import Disentangle_causal
from torch_geometric.utils import to_dense_adj, to_undirected
from gcn import GCN
import numpy as np 


class EncoderEmbedding(nn.Module):
    def __init__(self, q_num, concept_num,length,d_model):
        super(EncoderEmbedding, self).__init__()
        self.exercise_embed = nn.Embedding(q_num, d_model)
        self.response_embed = nn.Embedding(2, d_model)
        self.concept_embed = nn.Embedding(concept_num, d_model)

    def forward(self, exercises, pattern,response=None,concept=None,causal_mask=None, trivial_mask=None,QR_response=None):

        e = self.exercise_embed(exercises)

        if pattern=='only_q':
            c = self.concept_embed(concept)
            return e+c
        if pattern=='x_state':
            r= self.response_embed(response)
            c = self.concept_embed(concept)
            return e+r+c
        if pattern=='qc_embed':
            c = self.concept_embed(concept)
            return torch.cat([e,c],dim=-2) 
        if pattern=='x_reversal':
            c = self.concept_embed(concept)
            e=e+c
            r= self.response_embed(response)
            r2= self.response_embed(1-response) 
            return e.unsqueeze(1)+r.unsqueeze(1)*causal_mask.unsqueeze(-1)+r2.unsqueeze(1)*trivial_mask.unsqueeze(-1)
        if pattern=='q_replace':
            c = self.concept_embed(concept)
            r= self.response_embed(response)
            r=r+c
            e_replace = self.exercise_embed(QR_response)
            return r.unsqueeze(1)+e.unsqueeze(1)*causal_mask.unsqueeze(-1)+e_replace.unsqueeze(1)*trivial_mask.unsqueeze(-1)


class Predcit_head(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model=d_model
        self.out_fc = nn.Sequential(
            nn.Linear(int(self.d_model*2),int(self.d_model/4)), nn.ReLU(),
            nn.Linear(int(self.d_model/4),1),nn.Sigmoid())

    def forward(self, X, Q):
        h_pred=torch.cat([X,Q],dim=-1)
        out=self.out_fc(h_pred).squeeze(-1)
        return out

class Predcit_head_classifer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model=d_model
        self.out_fc = nn.Sequential(
            nn.Linear(int(self.d_model*2),int(self.d_model/4)), nn.ReLU(),
            nn.Linear(int(self.d_model/4),10))

    def forward(self, X, Q):
        h_pred=torch.cat([X,Q],dim=-1)
        out=self.out_fc(h_pred)
        return out

def get_fixed_edge(seq_len):
    q_source_nodes = list(range(0,int(seq_len-1)))
    q_target_nodes = list(range(1,seq_len))
    c_source_nodes = list(range(seq_len,int(seq_len+seq_len-1)))
    c_target_nodes = list(range(seq_len+1,seq_len+seq_len))
    source_nodes=q_source_nodes+c_source_nodes+q_source_nodes+c_source_nodes
    target_nodes=q_target_nodes+c_target_nodes+c_target_nodes+q_target_nodes
    qc_edge=torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return to_undirected(qc_edge)


class Model_exp(nn.Module):
    def __init__(self, q_num,concept_num, time_spend,d_model, length,nhead,num_encoder_layers, dropout,speed_cate,rate1,rate2):
        super(Model_exp, self).__init__()

        self.d_model=d_model
        seq_len=int(length-1)
        
        self.fixed_edge=to_dense_adj(get_fixed_edge(length)).cuda()
        self.gcn = GCN(
            d_model,
            d_model,
            d_model,
            num_layers=2,
            dropout=0.5)
        
        self.rate1=rate1
        self.rate2=rate2

        self.att_mask = torch.triu(torch.ones(seq_len, seq_len),diagonal=1).to(dtype=torch.bool).cuda()
        self.att_mask_num = 1-torch.triu(torch.ones(seq_len, seq_len),diagonal=1).cuda()

        self.encoder_embedding = EncoderEmbedding(q_num=q_num,concept_num=concept_num,
                                                  length=length, d_model=d_model)        

        self.x_encoder_num_layers=2
        self.q_encoder_num_layers=2
        self.rnn_q = nn.LSTM(d_model, d_model, num_layers=self.q_encoder_num_layers, batch_first=True)
        self.rnn = nn.LSTM(d_model, d_model, num_layers=self.x_encoder_num_layers, batch_first=True)

        self.Disentanglement=Disentangle_causal(hidden_size=d_model, seq_len=seq_len,tau=1, is_hard=True, dropout_p=0)

  #  ================ predict =====================
        self.pred_casual=Predcit_head(d_model)
        self.pred_trivial=Predcit_head_classifer(d_model)
        self.pred_interve=Predcit_head(d_model)
        self.pred_replace=Predcit_head(d_model)

        self.criterion = nn.BCELoss()
        self.criterion_fixed = nn.CrossEntropyLoss()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        """
        初始化
        """
        for p in self.parameters():
            # print(p)
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self,Q, Y, C,A,QR,S):
        Q=Q.int()
        Y=Y.int()
        S=S.int()
        C=C.int() 
        QR=QR.int()


        Q_predcit=Q[:,1:].contiguous()
        C_predcit=C[:,1:].contiguous()
        Y_true=Y[:,1:].contiguous()
        S_predict=S[:, 1:].contiguous()
        A_true=A[:,1:].contiguous()


        Q_response=Q[:,:-1].contiguous()
        Y_response=Y[:,:-1].contiguous()
        S_response=S[:,:-1].contiguous()
        C_response=C[:,:-1].contiguous()
        QR_response=QR[:,:-1].contiguous()


        batch_size,seq_len=Q_response.shape
        padding_response=S_response==0
        padding_predcit=S_predict==0

        X_embed = self.encoder_embedding(exercises=Q_response, response=Y_response,concept=C_response,pattern='x_state') # ([32, 199, 256]) 
        Q_embed = self.encoder_embedding(exercises=Q_predcit,concept=C_predcit,pattern='only_q') 

        C_node=torch.cat([C,C],dim=-1)
        dynamic_egde=((C_node.unsqueeze(-1)-C_node.unsqueeze(-2))==0).to(dtype=torch.int) 

        graph_egde=self.fixed_edge+dynamic_egde # batch*len*len + len*len
        QC_embed = self.encoder_embedding(exercises=Q,concept=C,pattern='qc_embed')
        qc_state=self.gcn(QC_embed, graph_egde)
        ques_state=qc_state[:,:int(seq_len+1),:]
        conc_state=qc_state[:,int(seq_len+1):,:]

        #  =====================================
        Q_state, _ = self.rnn_q(Q_embed, (torch.zeros(self.q_encoder_num_layers, batch_size, self.d_model).cuda(), torch.zeros(2, batch_size, self.d_model).cuda())) #future
        X_state, _ = self.rnn(X_embed, (torch.zeros(self.x_encoder_num_layers, batch_size, self.d_model).cuda(), torch.zeros(2, batch_size, self.d_model).cuda())) #future
        causal_mask, trivial_mask=self.Disentanglement(Q_state=Q_state, X_state=X_state,
                                                        att_mask=self.att_mask,
                                                        key_padding_mask=padding_response,
                                                        ques_state=ques_state,
                                                        conc_state=conc_state
                                                        ) #([32, 199, 199]) ([32, 199, 199])
    
        X_embed_causal=X_embed.unsqueeze(1)*causal_mask.unsqueeze(-1) #[32,1,199,256]*[32,199,199,1]-->([32, 199, 199, 512])
        X_embed_causal=X_embed_causal.view(batch_size*seq_len,seq_len,self.d_model)

        X_state_causal, _ = self.rnn(X_embed_causal, (torch.zeros(2, batch_size*seq_len, self.d_model).cuda(), torch.zeros(2, batch_size*seq_len, self.d_model).cuda())) #future

        X_state_causal=X_state_causal.view(batch_size,seq_len,seq_len,self.d_model)
        X_state_causal=X_state_causal*(self.att_mask_num.unsqueeze(0).unsqueeze(-1))#[32,199,199, 256]*[1, 199,199,1]-->[32, 199, 199, 256]
        X_causal_ave=torch.sum(X_state_causal,-2)/torch.sum(self.att_mask_num,-1).unsqueeze(-1)
    
        #  =====================================
        X_embed_trivial=X_embed.unsqueeze(1)*trivial_mask.unsqueeze(-1) #[32,1,199,256]*[32,199,199,1]-->([32, 199, 199, 512])
        X_embed_trivial=X_embed_trivial.view(batch_size*seq_len,seq_len,self.d_model)
        X_state_trivial, _ = self.rnn(X_embed_trivial, (torch.zeros(2, batch_size*seq_len, self.d_model).cuda(), torch.zeros(2, batch_size*seq_len, self.d_model).cuda())) #([32, 199, 199, 512])

        X_state_trivial=X_state_trivial.view(batch_size,seq_len,seq_len,self.d_model)
        
        X_state_trivial=X_state_trivial*(self.att_mask_num.unsqueeze(0).unsqueeze(-1))#[32,199,199, 256]*[1, 199,199,1]-->[32, 199, 199, 256]
        X_trivial_ave=torch.sum(X_state_trivial,-2)/torch.sum(self.att_mask_num,-1).unsqueeze(-1)

        #  =====================================

        X_embed_intervention = self.encoder_embedding(exercises=Q_response, response=Y_response,concept=C_response,
                                                      causal_mask=causal_mask, trivial_mask=trivial_mask,
                                                      pattern='x_reversal') #(([8, 199, 199, 512])

        X_embed_intervention = X_embed_intervention.view(batch_size*seq_len,seq_len,self.d_model)
        X_state_intervention, _ = self.rnn(X_embed_intervention,(torch.zeros(2, batch_size*seq_len, self.d_model).cuda(), torch.zeros(2, batch_size*seq_len, self.d_model).cuda())) #([32, 199, 199, 512])

        X_state_intervention = X_state_intervention.view(batch_size,seq_len,seq_len,self.d_model)

        X_state_intervention=X_state_intervention*(self.att_mask_num.unsqueeze(0).unsqueeze(-1))#[32,199,199, 256]*[1, 199,199,1]-->[32, 199, 199, 256]
        X_intervention_ave=torch.sum(X_state_intervention,-2)/torch.sum(self.att_mask_num,-1).unsqueeze(-1)

        #  =====================================

        X_embed_replace = self.encoder_embedding(exercises=Q_response, response=Y_response,concept=C_response,
                                                      causal_mask=causal_mask, trivial_mask=trivial_mask,
                                                      pattern='q_replace',QR_response=QR_response) #(([8, 199, 199, 512])

        X_embed_replace = X_embed_replace.view(batch_size*seq_len,seq_len,self.d_model)
        X_state_replace, _ = self.rnn(X_embed_replace, (torch.zeros(2, batch_size*seq_len, self.d_model).cuda(), torch.zeros(2, batch_size*seq_len, self.d_model).cuda())) #([32, 199, 199, 512])

        X_state_replace = X_state_replace.view(batch_size,seq_len,seq_len,self.d_model)

        X_state_replace=X_state_replace*(self.att_mask_num.unsqueeze(0).unsqueeze(-1))
        X_replace_ave=torch.sum(X_state_replace,-2)/torch.sum(self.att_mask_num,-1).unsqueeze(-1)

        #  =====================================

        X_causal_ave=self.norm1(X_causal_ave)
        X_trivial_ave=self.norm2(X_trivial_ave)
        X_intervention_ave=self.norm1(X_intervention_ave)
        X_replace_ave=self.norm1(X_replace_ave)

        y_causa=self.pred_casual(X_causal_ave,Q_state)
        y_trivi=self.pred_trivial(X_trivial_ave,Q_state)
        y_inter=self.pred_interve(X_intervention_ave,Q_state)
        y_repalce=self.pred_replace(X_replace_ave,Q_state)

        #  =====================================

        index = S_predict == 1

        loss_causa=self.criterion(y_causa[index], Y_true[index].float())
        loss_inter=self.criterion(y_inter[index], Y_true[index].float())
        loss_replace=self.criterion(y_repalce[index], Y_true[index].float())

        loss_trivi=self.criterion_fixed(y_trivi[index],A_true[index])

        loss_all=0.1*loss_causa+0.1*loss_inter+0.6*loss_trivi+0.3*loss_replace

        return loss_all,y_causa,Y_true,S_predict
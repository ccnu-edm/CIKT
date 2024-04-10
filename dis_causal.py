import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionScore_qc_weight(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        
        self.hidden_size=hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_q1 = nn.Linear(hidden_size, hidden_size)
        self.linear_k1 = nn.Linear(hidden_size, hidden_size)
        self.linear_q2 = nn.Linear(hidden_size, hidden_size)
        self.linear_k2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, ques_state, conc_state):
        q1=ques_state[:,1:].contiguous()
        k1=ques_state[:,:-1].contiguous()
        q2=conc_state[:,1:].contiguous()
        k2=conc_state[:,:-1].contiguous()
        q1=q1+q2
        k1=k1+k2

        q1 = self.linear_q1(q1)
        k1 = self.linear_k1(k1)
        scaling = float(self.hidden_size) ** -0.5
        attention1 = torch.bmm(q1, k1.transpose(-2, -1)) #([32, 199, 199])
        attention1 = attention1 * scaling
        attention=attention1
        return attention
    
class AttentionScore_causal(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        
        self.hidden_size=hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k,attn_mask,key_padding_mask,qc_score):
        q = self.linear_q(q)
        k = self.linear_k(k)
        attention = torch.bmm(q, k.transpose(-2, -1))
        scaling = float(self.hidden_size) ** -0.5
        attention = attention * scaling
        attention = attention * qc_score

        if attn_mask is not None:
            attention=attention.masked_fill(attn_mask,float('-inf')) 
        if key_padding_mask is not None:
            attention = attention.masked_fill(key_padding_mask.unsqueeze(1),float('-inf'))
        attention = self.softmax(attention)
        return attention

class AttentionScore_trivial(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        
        self.hidden_size=hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.FloatTensor([10]))
        self.b = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, q, k,attn_mask,key_padding_mask,qc_score):

        q = self.linear_q(q)
        k = self.linear_k(k)
        attention = torch.bmm(q, k.transpose(-2, -1)) 
        scaling = float(self.hidden_size) ** -0.5
        attention = attention * scaling
        attention = attention * qc_score

        attention=1-torch.sigmoid(attention) 
        if attn_mask is not None:
            attention=attention.masked_fill(attn_mask,float('-inf')) 
        if key_padding_mask is not None:
            attention = attention.masked_fill(key_padding_mask.unsqueeze(1),float('-inf'))
        attention = self.softmax(attention)
        return attention
    

class Disentangle_causal(nn.Module):
    def __init__(self, hidden_size, seq_len,tau=1, is_hard=True, dropout_p=0):
        super().__init__()
        self.tau = tau
        self.is_hard = is_hard
        self.causal_att = AttentionScore_causal(hidden_size, dropout_p)
        self.trivial_att = AttentionScore_trivial(hidden_size, dropout_p)
        self.qc_causal=AttentionScore_qc_weight(hidden_size, dropout_p)
        self.qc_trivial=AttentionScore_qc_weight(hidden_size, dropout_p)
        self.seq_len=seq_len


    def forward(self, Q_state, X_state, att_mask,key_padding_mask,ques_state,conc_state):

        qc_causal_score  = self.qc_causal(ques_state=ques_state, conc_state=conc_state) 
        qc_trivial_score = self.qc_trivial(ques_state=ques_state, conc_state=conc_state) 

        causal_score  = self.causal_att(q=Q_state, k=X_state, attn_mask = att_mask,key_padding_mask=key_padding_mask,qc_score=qc_causal_score) 
        trivial_score = self.trivial_att(q=Q_state, k=X_state, attn_mask = att_mask,key_padding_mask=key_padding_mask,qc_score=qc_trivial_score)
        score=torch.cat((causal_score.unsqueeze(2),trivial_score.unsqueeze(2)),2)

        score=F.gumbel_softmax(score, tau=self.tau, hard=self.is_hard, dim=2)
        causal_mask=score[:,:,0,:]
        trivial_mask=score[:,:,1,:]

        causal_mask = causal_mask.masked_fill(att_mask,0.) 
        trivial_mask=trivial_mask.masked_fill(att_mask,0.) 
        return causal_mask, trivial_mask


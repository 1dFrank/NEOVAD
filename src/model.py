
import torch
import torch.nn.init as torch_init
from layers import *
import torch.nn as nn
from configs_base2novel import build_config
import argparse
import clip

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)


class TemporalModule(nn.Module):
    '''
    Temporal Module 
    return the v_feature
    '''
    def __init__(self, cfg,d_model,n_heads, dropout_rate, gamma, bias, device,norm=None):
        super(TemporalModule, self).__init__()
        self.n_heads = n_heads
        self.self_attn = GATMultiHeadLayer(512,512//self.n_heads,dropout=dropout_rate,alpha=cfg.alpha,nheads=self.n_heads,concat=True)
        # self.self_attn2 = GATMultiHeadLayer(512,512//self.n_heads,dropout=dropout_rate,alpha=0.2,nheads=self.n_heads,concat=True)
        self.linear2 = nn.Linear(512,512)
        #self.linear1 = nn.Conv1d(d_model, 512, kernel_size=1) #512,the same as t_input
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        #self.norm = RMSNorm(d_model)
        self.device = device
        self.loc_adj = DistanceAdj(gamma, bias,self.device)
        # self.alpha = nn.Parameter(torch.tensor(0.))
        self.mask_rate = cfg.mask_rate
    def forward(self, x, seq_len=None):
        adj = self.loc_adj(x.shape[0], x.shape[1])#disadj:two version
        #simadj = self.adj(x, seq_len) #simadj 
         # mask the adj
        feats = x
        #print(feats.shape)
        feat_magnitudes = torch.norm(feats, p=2, dim=2)
        #print(feat_magnitudes.shape)
        k = int(self.mask_rate*feats.shape[1])# 0.4 
        topk = feat_magnitudes.topk(k, dim=-1).indices
        mask = torch.zeros_like(adj)
        for ix,i in enumerate(topk):
           mask[ix] =  mask[ix].index_fill(1,i,1)
           mask[ix] =  mask[ix].index_fill(0,i,1)    
        mask = mask.bool()
        adj = adj.masked_fill(~mask,0)

        tmp = self.self_attn(x, adj)
        # tmp_f = self.self_attn2(x,simadj)
        
      
        # tmp = self.alpha * tmp_f + (1 - self.alpha) * tmp_t
        
        if self.norm:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))  # power norm
            tmp = F.normalize(tmp)  # l2 norm

        x = x + self.linear2(tmp)
        
      
        x = self.norm(x).permute(0, 2, 1)
        # x = self.dropout1(F.gelu(self.linear1(x)))            
        return x
    # def adj(self, x, seq_len=None):
    #     # similarty adj
    #     soft = nn.Softmax(1)
    #     x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
    #     x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1

    #     x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
    #     x2 = x2 / (x_norm_x + 1e-20)
    #     output = torch.zeros_like(x2)
    #     if seq_len is None:
    #         for i in range(x.shape[0]):
    #             tmp = x2[i]
    #             adj2 = tmp
    #             adj2 = F.threshold(adj2, 0.5, 0)
    #             adj2 = soft(adj2)
                
    #             adj2 = F.threshold(adj2, 0.005, 0)
    #             output[i] = adj2
    #     else:
    #         
    #         for i in range(len(seq_len)):
    #             tmp = x2[i, :seq_len[i], :seq_len[i]]
    #             adj2 = tmp
    #             adj2 = F.threshold(adj2, 0.5, 0)
    #             adj2 = soft(adj2)
       
    #             adj2 = F.threshold(adj2, 0.005, 0)
    #             output[i, :seq_len[i], :seq_len[i]] = adj2

    #     return output


 
class Model(nn.Module):
    def __init__(self, cfg, clslist=None, vector_dict=None, token_dict=None,device='cpu'):
        super(Model, self).__init__()
        self.TM = TemporalModule(cfg,cfg.feat_dim,cfg.head_num, cfg.dropout_gat,cfg.gamma, cfg.bias,device)
       
        ###########ablation experiment：transformer
        # self.temporalModelling = TemporalModelling(width=512, layers=2, heads=4, dropout=0.6)
        ##############################

        self.device = device
        # notice to frozen its parameters
        self.clipmodel, _ = clip.load('ViT-B/16', device=self.device, jit=False) 
        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False
        # detector to [b,seqlen,1]
        self.classifier = nn.Sequential(
                nn.Conv1d(512, cfg.cls_hidden, kernel_size=1, padding=0),
                nn.GELU(),
                nn.Conv1d(cfg.cls_hidden,1,kernel_size=1,padding=0)
        ) 

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(512, 512 * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(512 * 4, 512))
        ]))   
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temp))
       
        # self.apply(weight_init)
        self.has_feature_input = cfg.has_feature_input
        self.temporal = cfg.temporal
        self.preprompt = cfg.preprompt
  
        self.promptpath = cfg.token_feat
        self.prefix = cfg.prefix
        self.adapter = cfg.adapter
        self.postfix = cfg.postfix
        self.clslist = clslist
        self.vector_dict = vector_dict
        self.token_dict = token_dict
        self.embedding = torch.nn.Embedding(77, 512)
        self.fixed_prompt = cfg.fixed_prompt
        self.norm = nn.LayerNorm(512)
        self.std_init = cfg.std_init
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=self.std_init)
        #torch_init.xavier_uniform_(self.embedding.weight)

    def encode_learnable_prompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.embedding(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prefix + 1: self.prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prefix + ind + self.postfix] = word_embedding[i, ind]
           ## add xct
            text_tokens[i, 0] = word_tokens[i, 0]
            text_tokens[i, self.prefix + 1: self.prefix + ind] = word_tokens[i, 1: ind]
            ##
            text_tokens[i, self.prefix + ind + self.postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def fixed_learnable_prompt(self, text):
        text = ['a video from a CCTV camera of a '+i for i in text]
        
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_features = self.clipmodel.encode_text(word_embedding,word_tokens)

        return text_features
    def forward(self, x, seq_len=None,clslist=None):
        
        #video feature
        if self.has_feature_input:
            x_v = x
        else: 
            #directly from CLIP image encoder
            pass
        if self.temporal:
            x_v = self.TM(x_v, seq_len) #in:[b,t,512];out:[b,512,t]
           
            ####ablation experiment：test transformer#####
            # x_v = self.temporalModelling(x_v).permute(0,2,1)
            ################################
        else:
            x_v = F.normalize(x_v, dim=-1)
            x_v = x_v.permute(0,2,1)
        logits = self.classifier(x_v)
        logits = logits.permute(0, 2, 1)
        logits = torch.sigmoid(logits)
        vFeature = x_v.permute(0,2,1)

      
        t_feature_pre = torch.from_numpy(np.load(self.promptpath)).to(self.device)
        if not self.fixed_prompt: #learnable prompt
            t_feature_le = self.encode_learnable_prompt(clslist)
        else: 
            t_feature_le = self.fixed_learnable_prompt(clslist)

            # encode text
           
                #  text embedding from prompting: base on the input cls_dict
                # embedding order follows the values of cls_name(to match the multi_label(id of cls))
     
        return logits, vFeature,t_feature_pre,t_feature_le
        
        ## +visual feature
        # logits_attn = logits.permute(0, 2, 1)
        # visual_attn = logits_attn @ vFeature
        # visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        # visual_attn = visual_attn.expand(visual_attn.shape[0], t_feature_pre.shape[0], visual_attn.shape[2])
        # text_features = t_feature_pre.unsqueeze(0)
        # text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
      
        # t_feature_pre = text_features + visual_attn

        
       
        # return logits, vFeature,t_feature_pre,t_feature_le

class SimpleAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // 4, dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class AdaptedModel(nn.Module):
    def __init__(self, cfg, clslist=None, vector_dict=None, token_dict=None,device='cpu'):
        super(AdaptedModel, self).__init__()
        
        
        self.TM = TemporalModule(cfg,cfg.feat_dim,cfg.head_num, cfg.dropout_gat,cfg.gamma, cfg.bias,device)
        
        # self.TM = TemporalModelling(width=512, layers=2, heads=4, dropout=0.6)
        self.device = device
        # notice to frozen its parameters
        self.clipmodel, _ = clip.load('ViT-B/16', device=self.device, jit=False) 
        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False
        # detector to [b,seqlen,1]
        self.classifier = nn.Sequential(
                nn.Conv1d(512, cfg.cls_hidden, kernel_size=1, padding=0),
                nn.GELU(),
                nn.Conv1d(cfg.cls_hidden,1,kernel_size=1,padding=0)
        ) 

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(512, 512 * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(512 * 4, 512))
        ]))   
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temp))
       
        # text adapter
        self.text_adapt_until = cfg.text_adapt_until
        self.text_adapt_weight = cfg.text_adapt_weight
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(self.clipmodel.transformer.width) for _ in range(self.text_adapt_until)]
        )
        # self.apply(weight_init)
        self.has_feature_input = cfg.has_feature_input
        self.temporal = cfg.temporal
        self.adapter = cfg.adapter
        self.preprompt = cfg.preprompt
  
        self.promptpath = cfg.token_feat
        self.prefix = cfg.prefix
        self.postfix = cfg.postfix
        self.clslist = clslist
        self.vector_dict = vector_dict
        self.token_dict = token_dict
        self.embedding = torch.nn.Embedding(77, 512)
        self.fixed_prompt = cfg.fixed_prompt
        self.norm = nn.LayerNorm(512)
        self.std_init = cfg.std_init
        self.seed = cfg.seed
        self.initialize_parameters()
        

    def initialize_parameters(self):
        # torch.manual_seed(self.seed)
        nn.init.normal_(self.embedding.weight, std=self.std_init)
        # for adapter in self.text_adapter:
        #     nn.init.xavier_uniform_(adapter.fc1.weight)
        #     nn.init.zeros_(adapter.fc1.bias)
        #     nn.init.xavier_uniform_(adapter.fc2.weight)
        #     nn.init.zeros_(adapter.fc2.bias)
        # torch_init.xavier_uniform_(self.embedding.weight)

    def encode_learnable_prompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.embedding(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prefix + 1: self.prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prefix + ind + self.postfix] = word_embedding[i, ind]
           ## add xct
            text_tokens[i, 0] = word_tokens[i, 0]
            text_tokens[i, self.prefix + 1: self.prefix + ind] = word_tokens[i, 1: ind]
            ##
            text_tokens[i, self.prefix + ind + self.postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        # text_features = self.encode_text_with_adapter(text_embeddings)

        all_prompts = [f"a {cls_name} video from a CCTV camera" for cls_name in text]
        abnormal_prompts = all_prompts[1:] 
        anomaly_tokens = clip.tokenize(abnormal_prompts).to(self.device)
        anomaly_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
        

        normal_prompts = [f"a normal video without {cls_name}" for cls_name in text[1:]]

        normal_tokens = clip.tokenize(normal_prompts).to(self.device)
        normal_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)

        pair_features = torch.stack([normal_features, anomaly_features], dim=1)

        return text_features, pair_features

    def fixed_learnable_prompt(self, text):
        
        all_prompts = [f"a {cls_name} video from a CCTV camera" for cls_name in text]

        
        if not self.adapter:
            '''no text adapter'''
            # print(" no Using adapter.")
            all_tokens = clip.tokenize(all_prompts).to(self.device)
            all_embeddings = self.clipmodel.encode_token(all_tokens)
            all_features = self.clipmodel.encode_text(all_embeddings, all_tokens)

            all_features = all_features / all_features.norm(dim=-1, keepdim=True)

            abnormal_prompts = all_prompts[1:] 
            anomaly_tokens = clip.tokenize(abnormal_prompts).to(self.device)
            anomaly_embeddings = self.clipmodel.encode_token(anomaly_tokens)
            anomaly_features = self.clipmodel.encode_text(anomaly_embeddings, anomaly_tokens)
            anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
            

            normal_prompts = [f"a normal video without {cls_name}" for cls_name in text[1:]]

            normal_tokens = clip.tokenize(normal_prompts).to(self.device)
            normal_embeddings = self.clipmodel.encode_token(normal_tokens)
            normal_features = self.clipmodel.encode_text(normal_embeddings, normal_tokens)
            normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)

            pair_features = torch.stack([normal_features, anomaly_features], dim=1)
        else:
            '''using text adapter'''
            # print("Using adapter.")
            all_tokens = clip.tokenize(all_prompts).to(self.device)
            all_features = self.encode_text_with_adapter(all_tokens)
            all_features = all_features / all_features.norm(dim=-1, keepdim=True)

            abnormal_prompts = all_prompts[1:] 
            anomaly_tokens = clip.tokenize(abnormal_prompts).to(self.device)
            anomaly_features = self.encode_text_with_adapter(anomaly_tokens)
            anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
            
            
            normal_prompts = [f"a normal video without {cls_name}" for cls_name in text[1:]]

            normal_tokens = clip.tokenize(normal_prompts).to(self.device)
            normal_features = self.encode_text_with_adapter(normal_tokens)
            normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
            
            pair_features = torch.stack([normal_features, anomaly_features], dim=1)

        return all_features, pair_features
    
    def fixed_learnable_prompt_new(self, text):

        all_prompts = [f"a {cls_name} video from a CCTV camera" for cls_name in text]
        all_tokens = clip.tokenize(all_prompts).to(self.device)
        all_features = self.encode_text_with_adapter(all_tokens)
        all_features = all_features / all_features.norm(dim=-1, keepdim=True)

        origin_abnormal_prompts = all_prompts[1:]
        origin_anomaly_tokens = clip.tokenize(origin_abnormal_prompts).to(self.device)
        origin_anomaly_features = self.encode_text_with_adapter(origin_anomaly_tokens)
        origin_anomaly_features = origin_anomaly_features / origin_anomaly_features.norm(dim=-1, keepdim=True)

        TARGET_CLS = 11

    
        orig_cls_sentences = all_prompts[1:]  
        orig_cls_labels = text[1:]            

        cls_num = len(orig_cls_sentences)
        if cls_num == 0:
            raise ValueError("No abnormal classes found in text (text[1:] is empty).")

        
        tokens = clip.tokenize(orig_cls_sentences).to(self.device)
        feats = self.encode_text_with_adapter(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        mean_feat = feats.mean(dim=0, keepdim=True)
        scores = ((feats - mean_feat) ** 2).sum(dim=-1)  # (cls_num,)

        
        device = feats.device
        if cls_num >= TARGET_CLS:
            _, topk_idx = torch.topk(scores, TARGET_CLS, largest=True)
            selected_indices = topk_idx
        else:
            
            order_desc = torch.argsort(scores, descending=True)
            need = TARGET_CLS - cls_num

            base_indices = torch.arange(cls_num, device=device)

            if need <= cls_num:
                extra = order_desc[:need]
            else:
                repeats = (need + cls_num - 1) // cls_num
                extra = order_desc.repeat(repeats)[:need]

            selected_indices = torch.cat([base_indices, extra], dim=0)

       
        selected_sentences = [orig_cls_sentences[i] for i in selected_indices.tolist()]
        selected_labels = [orig_cls_labels[i] for i in selected_indices.tolist()]

      
        ab_tokens = clip.tokenize(selected_sentences).to(self.device)
        ab_feats = self.encode_text_with_adapter(ab_tokens)
        ab_feats = ab_feats / ab_feats.norm(dim=-1, keepdim=True)

        # normal prompt 
        normal_prompts = [f"a normal video without {lbl}" for lbl in selected_labels]
        nor_tokens = clip.tokenize(normal_prompts).to(self.device)
        nor_feats = self.encode_text_with_adapter(nor_tokens)
        nor_feats = nor_feats / nor_feats.norm(dim=-1, keepdim=True)

        # (11, 2, D)
        pair_features = torch.stack([nor_feats, ab_feats], dim=1)

        return all_features, pair_features, origin_anomaly_features


    
    
    def encode_text_with_adapter(self, text, cast_dtype=None):
        x = self.clipmodel.token_embedding(text)
        if cast_dtype is not None:
            x = x.to(cast_dtype)

        x = x + self.clipmodel.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)  # LND

        # --- Transformer  ---
        for i in range(len(self.clipmodel.transformer.resblocks)):
            x= self.clipmodel.transformer.resblocks[i](x)

            # adapter tuning 
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)

                
                adapt_out = (
                    adapt_out *
                    x.norm(dim=-1, keepdim=True) /
                    adapt_out.norm(dim=-1, keepdim=True)
                )

                
                x = self.text_adapt_weight * adapt_out + (1 - self.text_adapt_weight) * x

        # --- LN & EOT token ---
        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        return x

    def forward(self, x, seq_len=None,clslist=None):
        
        #video feature
        if self.has_feature_input:
            x_v = x
        else: 
            #directly from CLIP image encoder
            pass
        if self.temporal:
            x_v = self.TM(x_v, seq_len) #in:[b,t,512];out:[b,512,t]
           
            ####ablation experiment：test transformer#####
            # x_v = self.temporalModelling(x_v).permute(0,2,1)
            ################################
        else:
            # x_v = self.temporalModelling(x_v).permute(0,2,1)
            x_v = F.normalize(x_v, dim=-1)
            x_v = x_v.permute(0,2,1)
        logits = self.classifier(x_v)
        logits = logits.permute(0, 2, 1)
        logits = torch.sigmoid(logits)
        vFeature = x_v.permute(0,2,1)


        '''return revised orthogonal loss'''
        # pair_features = None
        # origin_abnormal_features = None
        # t_feature_pre = torch.from_numpy(np.load(self.promptpath)).to(self.device)
        # if not self.fixed_prompt: #learnable prompt
        #     t_feature_le, pair_features = self.encode_learnable_prompt(clslist)
        # else: 
        #     t_feature_le, pair_features, origin_abnormal_features = self.fixed_learnable_prompt(clslist)
     
        # return logits, vFeature, t_feature_pre, t_feature_le, pair_features, origin_abnormal_features


        '''return simple orthogonal loss'''
        pair_features = None
        t_feature_pre = torch.from_numpy(np.load(self.promptpath)).to(self.device)
        if not self.fixed_prompt: #learnable prompt
            t_feature_le, pair_features = self.encode_learnable_prompt(clslist)
        else: 
            t_feature_le, pair_features = self.fixed_learnable_prompt(clslist)
     
        return logits, vFeature, t_feature_pre, t_feature_le, pair_features
        
        ## +visual feature
        # logits_attn = logits.permute(0, 2, 1)
        # visual_attn = logits_attn @ vFeature
        # visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        # visual_attn = visual_attn.expand(visual_attn.shape[0], t_feature_pre.shape[0], visual_attn.shape[2])
        # text_features = t_feature_pre.unsqueeze(0)
        # text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
      
        # t_feature_pre = text_features + visual_attn

        
       
        # return logits, vFeature,t_feature_pre,t_feature_le
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAD')
    args = parser.parse_args()
    cfg = build_config('ucf')

    model = Model(cfg)
    x = torch.randn(128,200,512)
    li = list(open(r"list\prompt\ucf_cls.txt"))
    clslist = [i.strip().lower() for i in li]
    print(clslist)
    logits, x_v,t_feature_pre,t_feature_le = model(x,seq_len=200,clslist=clslist)
    print(model)
    print(logits.shape,x_v.shape)
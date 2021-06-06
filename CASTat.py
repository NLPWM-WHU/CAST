import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CASTat(nn.Module):
    def __init__(self, num_users, num_items, num_categoies, num_tims, model_args):
        super(CASTat, self).__init__()
        self.args = model_args

        # init args
        dims = self.args.d
        self.dims = dims
        self.drop_ratio = self.args.drop
        self.hidden_units = dims
        # user and item and category embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)
        self.time_embeddings = nn.Embedding(num_tims, dims)
        self.category_embeddings = nn.Embedding(num_categoies, dims)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.time_embeddings.weight.data.normal_(0, 1.0 / self.time_embeddings.embedding_dim)
        self.category_embeddings.weight.data.normal_(0, 1.0 / self.category_embeddings.embedding_dim)

        # RNN
        self.encoder = nn.LSTM(input_size=dims, hidden_size=dims, bidirectional=False)
        self.encoder1 = nn.LSTM(input_size=dims * 2, hidden_size=dims, bidirectional=False)
        self.encoder2 = nn.LSTM(input_size=dims * 2, hidden_size=dims, bidirectional=False)
        self.encoder3 = nn.LSTM(input_size=dims * 2, hidden_size=dims, bidirectional=False)
        self.encoder4 = nn.LSTM(input_size=dims * 2, hidden_size=dims, bidirectional=False)

        self.linear_50_50 = nn.Linear(dims, dims)
        self.linear_50_50_1 = nn.Linear(dims, dims)
        self.linear_50_50_2 = nn.Linear(dims * 2, dims * 2)
        self.out_item = nn.Linear(int(dims/2), num_items)
        self.out_item1 = nn.Linear(int(dims/2), num_items)
        self.out_item2 = nn.Linear(dims, num_items)
        self.out_cate = nn.Linear(dims, num_categoies)
        self.out_cate1 = nn.Linear(dims, num_categoies)
        self.change = nn.Linear(int(dims/2), dims)
        self.change1 = nn.Linear(int(dims/2), dims)
        nn.init.xavier_normal_(self.linear_50_50.weight)
        nn.init.xavier_normal_(self.out_item.weight)
        nn.init.xavier_normal_(self.out_item1.weight)
        nn.init.xavier_normal_(self.out_item2.weight)
        nn.init.xavier_normal_(self.out_cate.weight)
        nn.init.xavier_normal_(self.out_cate1.weight)
        nn.init.xavier_normal_(self.change.weight)
        nn.init.xavier_normal_(self.change1.weight)
        nn.init.xavier_normal_(self.linear_50_50_1.weight)
        nn.init.xavier_normal_(self.linear_50_50_2.weight)
        self.activation = nn.ReLU()
        self.softplus = nn.Softplus()


    def forward(self, seq_var, user_var, item_var, tim_seq_var, cat_seq_var, for_pred=False):
        # batch size
        batch_size = seq_var.size()[0]
        sequence_size = seq_var.size()[1]
        # Embedding Look-up
        item_embs = self.dropout(self.item_embeddings(seq_var))   # use unsqueeze() to get 3-D
        categories_embs = self.dropout(self.category_embeddings(cat_seq_var))
        tims_embs = self.dropout(self.time_embeddings(tim_seq_var))
        user_emb = self.dropout(self.user_embeddings(user_var))
        item_embs = item_embs.transpose(0, 1)
        categories_embs = categories_embs.squeeze(2).transpose(0, 1)
        tims_embs = tims_embs.squeeze(2).transpose(0, 1)
        user_emb_r = user_emb.repeat(1, sequence_size, 1).transpose(0, 1)


        ### category information
        ### Layer 1
        h = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        c = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        outputs_hi_1, (_,_) = self.encoder(item_embs, (h,c)) # outputs_hi=>h^{(1)}_{i_k}

        ### VAE 1
        outputs_hi_1 = self.linear_50_50(outputs_hi_1)
        mu = outputs_hi_1[:,:,int(self.dims/2):]
        log_sigma = self.softplus(outputs_hi_1[:,:,:int(self.dims/2)])
        if for_pred:
            sampled_z = mu
        else:
            sigma = torch.exp(0.5 * log_sigma)
            std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()
            # Reparameterization trick
            sampled_z = mu + sigma * Variable(std_z, requires_grad=False) # sampled_z=>z_k

        ### Layer 2
        h = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        c = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        output_hc_2, (_, _) = self.encoder1(torch.cat((categories_embs, self.change(sampled_z)), dim=2), (h,c)) # output_hc=>h^{(2)}_{c_k}
        item_predict = self.out_item(sampled_z).transpose(0, 1)
        item_predict = F.log_softmax(item_predict, -1) # for L^{(1)}_i
        cate_predict1 = self.out_cate1(output_hc_2).transpose(0, 1)
        cate_predict1 = F.log_softmax(cate_predict1, -1) # for L^{(2)}_c


        ### Layer 3 (also the layer 1 for time context)
        ### note that the output of Layer 3 is used as the input of VAE for time context
        h = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        c = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        outputs_hi_3, (_, _) = self.encoder3(torch.cat((item_embs, output_hc_2), dim = 2), (h, c))
        outputs_hi_3 = self.linear_50_50_1(outputs_hi_3)
        mu_2 = outputs_hi_3[:, :, int(self.dims/2):]
        log_sigma_2 = self.softplus(outputs_hi_3[:, :, :int(self.dims/2)])
        if for_pred:
            sampled_z_2 = mu_2
        else:
            sigma_2 = torch.exp(0.5 * log_sigma_2)
            std_z_2 = torch.from_numpy(np.random.normal(0, 1, size=sigma_2.size())).float().cuda()
            sampled_z_2 = mu_2 + sigma_2 * Variable(std_z_2, requires_grad=False)
        item_predict1 = self.out_item1(sampled_z_2).transpose(0, 1)
        item_predict1 = F.log_softmax(item_predict1, -1)


        ### Layer 2 for time context
        h = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        c = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        output_ht_2, _ = self.encoder2(torch.cat((tims_embs, self.change1(sampled_z_2)), dim=2), (h, c))


        ### Layer 3 for time context
        h = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        c = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        output_last, _ = self.encoder4(torch.cat((item_embs, output_ht_2), dim=2), (h, c))


        ### PFP
        output_last = torch.cat((output_last, user_emb_r), dim=2) # output_last=>h_{p_k}
        output_last = self.linear_50_50_2(output_last)
        mu_3 = output_last[:, :, self.dims:]
        log_sigma_3 = self.softplus(output_last[:, :, :self.dims])
        if for_pred:
            sampled_z_3 = mu_3
        else:
            sigma_3 = torch.exp(0.5 * log_sigma_3)
            std_z_3 = torch.from_numpy(np.random.normal(0, 1, size=sigma_3.size())).float().cuda()
            sampled_z_3 = mu_3 + sigma_3 * Variable(std_z_3, requires_grad=False) # sampled_z_3=>z_{p_k}

        item_predict2 = self.out_item2(sampled_z_3).transpose(0, 1)
        item_predict2 = F.log_softmax(item_predict2, -1) # for L^{(f)}

        #kld value
        kld = torch.mean(torch.sum(0.5 * (-log_sigma + torch.exp(log_sigma) + mu ** 2 - 1), -1))## keep same with svae
        kld2 = torch.mean(torch.sum(0.5 * (-log_sigma_2 + torch.exp(log_sigma_2) + mu_2 ** 2 - 1), -1))  ## keep same with svae
        kld3 = torch.mean(torch.sum(0.5 * (-log_sigma_3 + torch.exp(log_sigma_3) + mu_3 ** 2 - 1), -1))  ## keep same with svae

        if for_pred:
            item_predict_last = item_predict2[:,-1,:]
            item_var = item_var.squeeze(2)
            res = torch.gather(item_predict_last, 1, item_var)
        else:
            res = item_predict

        return res, kld, 0, item_predict1, kld2, item_predict2, kld3, cate_predict1

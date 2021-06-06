import argparse
from time import time
import logging
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from CASTat import CASTat
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import *
import os
import copy

DATASET_NAME = ''
class Recommender(object):
    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 learning_rate=None,
                 use_cuda=False,
                 kld1=None,
                 kld2=None,
                 kld3=None,
                 model_args=None):

        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._device = torch.device("cuda" if use_cuda else "cpu")

        # rank evaluation related
        self.test_sequence = None
        self.valid_sequence = None
        self._candidate = dict()
        self.kld1 = kld1
        self.kld2 = kld2
        self.kld3 = kld3

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_categories = interactions.num_categories
        self._num_users = interactions.num_users
        self._num_tims = 24
        self.test_sequence = interactions.test_sequences
        self.valid_sequence = interactions.valid_sequences
        self._net = CASTat(self._num_users,
                         self._num_items,
                         self._num_categories,
                         self._num_tims,
                         self.model_args).to(self._device)
        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)
    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def fit(self, train, verbose=False):
        # convert to sequences, targets and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)
        n_train = sequences_np.shape[0]
        output_str = 'total training instances: %d' % n_train
        print(output_str)

        if not self._initialized:
            self._initialize(train)
        start_epoch = 0
        #####
        for epoch_num in range(start_epoch, self._n_iter):
            if epoch_num <= 3:
                k1 = 0
                k2 = 0
                k3 = 0
            else:
                k1 = self.kld1
                k2 = self.kld2
                k3 = self.kld3

            t1 = time()

            # set model to training mode
            self._net.train()

            users_np, sequences_np, targets_np = shuffle(users_np,
                                                         sequences_np,
                                                         targets_np)
            users, sequences, targets = (torch.from_numpy(users_np).long(),
                                         torch.from_numpy(sequences_np).long(),
                                         torch.from_numpy(targets_np).long())

            users, sequences, targets = (users.to(self._device),
                                         sequences.to(self._device),
                                         targets.to(self._device))

            epoch_loss = 0.0
            kld_loss = 0.0
            kld2_loss = 0.0
            kld3_loss = 0.0
            seq2seq_item_loss = 0.0
            seq2seq_cate_loss = 0.0
            seq2seq_item1_loss = 0.0
            seq2seq_item2_loss = 0.0
            seq2seq_cate2_loss = 0.0
            for (minibatch_num,
                 (batch_users,
                  batch_sequences,
                  batch_targets
                  )) in enumerate(minibatch(users,
                                            sequences,
                                            targets,
                                            batch_size=self._batch_size)):
                batch_item_sequences = batch_sequences[:, :, 0].to(self._device)
                batch_time_sequences = batch_sequences[:, :, 1].to(self._device)
                batch_cate_sequences = batch_sequences[:, :, 2].to(self._device)

                batch_item_sequences_t = batch_targets[:, :, 0].to(self._device)
                batch_time_sequences_t = batch_targets[:, :, 1].to(self._device)
                batch_cate_sequences_t = batch_targets[:, :, 2].to(self._device)
                batch_time_sequences_input = torch.cat((batch_time_sequences[:, 1:], batch_time_sequences_t), dim=-1)
                items_to_predict = batch_item_sequences_t
                item_predict, kld, cate_predict, item_predict1, kld2, item_predict2, kld3, cate_predict2 = self._net(
                                                                                                                    batch_item_sequences,
                                                                                                                    batch_users,
                                                                                                                    items_to_predict,
                                                                                                                    batch_time_sequences_input,
                                                                                                                    batch_cate_sequences)
                self._optimizer.zero_grad()
                item_for_prediction_encoder = torch.cat((batch_item_sequences, batch_item_sequences_t), dim=1)[:,1:].unsqueeze(2)
                cate_for_prediction_encoder = torch.cat((batch_cate_sequences, batch_cate_sequences_t), dim=1)[:,1:].unsqueeze(2)
                likelihood = (-1.0) * torch.sum(torch.gather(item_predict, 2, item_for_prediction_encoder))
                likelihood1 = (-1.0) * torch.sum(torch.gather(item_predict1, 2, item_for_prediction_encoder))
                likelihood2 = (-1.0) * torch.sum(torch.gather(item_predict2, 2, item_for_prediction_encoder))
                likelihood_c2 = (-1.0) * torch.sum(torch.gather(cate_predict2, 2, cate_for_prediction_encoder))

                loss = likelihood + likelihood_c2 + likelihood1 + likelihood2 + k1 * kld + k2 * kld2 +k3 * kld3
                # L^(1)_i + L^(2)_c + L^(1)_i +L^(f)_i + \lambda*{KL^{(2)}_c + KL^{(2)}_c + KL^{(f)}_i}

                kld_loss += (kld).item()
                kld2_loss += (kld2).item()
                seq2seq_item_loss += likelihood.item()
                seq2seq_item1_loss += likelihood1.item()
                kld3_loss += (kld3).item()
                seq2seq_item2_loss += likelihood2.item()
                seq2seq_cate2_loss += likelihood_c2.item()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1
            kld_loss /= minibatch_num + 1
            kld2_loss /= minibatch_num + 1
            kld3_loss /= minibatch_num + 1
            seq2seq_item_loss /= minibatch_num + 1
            seq2seq_cate_loss /= minibatch_num + 1
            seq2seq_item1_loss /= minibatch_num + 1
            seq2seq_item2_loss /= minibatch_num + 1
            seq2seq_cate2_loss /= minibatch_num + 1
            t2 = time()
            if verbose and (epoch_num + 1) % 1 == 0:
                loss_str = "Epoch %d [%.1f s]\tepochloss=%.4f, kld_loss=%.4f, kld2_loss=%.4f, kld3_loss=%.4f,seq2seqitemloss=%.4f,seq2seq_cate_loss = %.4f, " \
                           "seq2seq_item1_loss = %.4f, seq2seq_item2_loss=%.4f, seq2seq_cate2_loss=%.4f" \
                           % (epoch_num + 1,
                              t2 - t1,
                              epoch_loss,
                              kld_loss,
                              kld2_loss,
                              kld3_loss,
                              seq2seq_item_loss,
                              seq2seq_cate_loss,
                              seq2seq_item1_loss,
                              seq2seq_item2_loss,
                              seq2seq_cate2_loss)
                NDCG, HT, F1, AUC = evaluate_ranking(self, train, k=[1, 5, 10], is_valid=False)
                NDCGv, HTv, F1v, AUCv = evaluate_ranking(self, train, k=[1, 5, 10], is_valid=True)
                output_str = "NDCG@1=%.4f, NDCG@5=%.4f, NDCG@10=%.4f, NDCG@15=%.4f, NDCG@20=%.4f, " \
                             "HT@1=%.4f, HT@5=%.4f, HT@10=%.4f, HT@15=%.4f, HT@20=%.4f, " \
                             "F1@1=%.4f, F1@5=%.4f, F1@10=%.4f, F1@15=%.4f, F1@20=%.4f, AUC=%.4f, [%.1f s]" % (
                                 NDCG[0],
                                 NDCG[1],
                                 NDCG[2],
                                 NDCG[3],
                                 NDCG[4],
                                 HT[0],
                                 HT[1],
                                 HT[2],
                                 HT[3],
                                 HT[4],
                                 F1[0],
                                 F1[1],
                                 F1[2],
                                 F1[3],
                                 F1[4],
                                 AUC,
                                 time() - t2)
                output_strv = "NDCG@1=%.4f, NDCG@5=%.4f, NDCG@10=%.4f, NDCG@15=%.4f, NDCG@20=%.4f, " \
                              "HT@1=%.4f, HT@5=%.4f, HT@10=%.4f, HT@15=%.4f, HT@20=%.4f, " \
                              "F1@1=%.4f, F1@5=%.4f, F1@10=%.4f, F1@15=%.4f, F1@20=%.4f, AUC=%.4f, [%.1f s]" % (
                              NDCGv[0],
                              NDCGv[1],
                              NDCGv[2],
                              NDCGv[3],
                              NDCGv[4],
                              HTv[0],
                              HTv[1],
                              HTv[2],
                              HTv[3],
                              HTv[4],
                              F1v[0],
                              F1v[1],
                              F1v[2],
                              F1v[3],
                              F1v[4],
                              AUCv,
                              time() - t2)
                print(loss_str)
                print("test:")
                print(output_str)
                print("valid:")
                print(output_strv)
                logging.info(loss_str)
                logging.info("test:")
                logging.info(output_str)
                logging.info("valid:")
                logging.info(output_strv)
                torch.save(self._net, "datasets/{}/CASTat_".format(DATASET_NAME) + str(epoch_num + 1) + ".model")
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (epoch_num + 1,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                print(output_str)
                logging.info(output_str)

    def create_batch_predict(self, user_ids, item_ids, valid_flag):
        sequences_batch = []
        user_batch = []
        target_c_batch = []
        sequences_c_batch = []
        target_t_batch = []
        sequences_t_batch = []
        items_batch = []
        for user_id, item_id in zip(user_ids, item_ids):
            if valid_flag == False:
                sequences_np = self.test_sequence.sequences[user_id, :]
                target_np = self.test_sequence.targets[user_id, :]
            elif valid_flag == True:
                sequences_np = self.valid_sequence.sequences[user_id, :]
                target_np = self.valid_sequence.targets[user_id, :]

            sequences_np = np.atleast_2d(sequences_np)

            if item_id is None:
                item_id = np.arange(self._num_items).reshape(-1, 1)
            batch_item_sequences = sequences_np[:, 0]
            batch_time_sequences = sequences_np[:, 1]
            batch_cate_sequences = sequences_np[:, 2]

            batch_item_sequences_t = target_np[:, 0]
            batch_time_sequences_t = target_np[:, 1]
            batch_cate_sequences_t = target_np[:, 2]
            batch_time_sequences = np.append(batch_time_sequences[1:], batch_time_sequences_t)

            items = torch.from_numpy(item_id).long()
            user = torch.from_numpy(np.array([[user_id]])).long()
            sequences_batch.append(batch_item_sequences)
            user_batch.append(user)
            items_batch.append(items)
            target_t_batch.append(batch_time_sequences_t)
            sequences_t_batch.append(batch_time_sequences)
            target_c_batch.append(batch_cate_sequences_t)
            sequences_c_batch.append(batch_cate_sequences)

        sequences_batch = np.stack(sequences_batch, axis=0)
        user_batch = np.stack(user_batch, axis=0)
        items_batch = np.stack(items_batch, axis=0)
        target_c_batch = np.stack(target_c_batch, axis=0)
        sequences_c_batch = np.stack(sequences_c_batch, axis=0)
        target_t_batch = np.stack(target_t_batch, axis=0)
        sequences_t_batch = np.stack(sequences_t_batch, axis=0)

        sequences_batch = torch.from_numpy(sequences_batch).to(self._device).squeeze(1)
        user_batch = torch.from_numpy(user_batch).to(self._device).squeeze(1)
        items_batch = torch.from_numpy(items_batch).to(self._device).squeeze(1)
        target_c_batch = torch.from_numpy(target_c_batch).long().to(self._device)
        sequences_c_batch = torch.from_numpy(sequences_c_batch).long().to(self._device).squeeze(1)
        target_t_batch = torch.from_numpy(target_t_batch).long().to(self._device)
        sequences_t_batch = torch.from_numpy(sequences_t_batch).long().to(self._device).squeeze(1)

        return sequences_batch, user_batch, target_c_batch, sequences_c_batch, items_batch, target_t_batch, sequences_t_batch

    def predict(self, user_id, item_ids=None, valid_flag=False):
        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')

        # set model to evaluation model
        self._net.eval()
        with torch.no_grad():
            sequences_batch, user_batch, target_c_batch, sequences_c_batch, items_batch, target_t_batch, sequences_t_batch = self.create_batch_predict(
                user_id, item_ids, valid_flag)

            out, _, _, _, _, _,_ ,_ = self._net(sequences_batch,
                                              user_batch,
                                              items_batch,
                                              sequences_t_batch,
                                              sequences_c_batch,
                                              for_pred=True)
        return out.cpu().numpy()

def set_dataname(dataset):
    global DATASET_NAME
    DATASET_NAME = dataset

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gowalla')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--Window', type=int, default=500)
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--kld1', type=float, default=10)
    parser.add_argument('--kld2', type=float, default=10)
    parser.add_argument('--kld3', type=float, default=10)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='0')

    config = parser.parse_args()
    set_dataname(config.dataset)
    dataset_path = "datasets/{}/{}.txt".format(config.dataset, config.dataset)
    log_file_name = "datasets/{}/{}_r{}_CASTat_b5_l{}_bs{}_W{}_kld1{}_kld2{}_kld3{}.txt".format(config.dataset,
                                                                                                config.dataset,
                                                                                                config.seed,
                                                                                                config.L,
                                                                                                config.batch_size,
                                                                                                config.Window,
                                                                                                config.kld1,
                                                                                                config.kld2,
                                                                                                config.kld3)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device
    test_random_path = "datasets/{}/test_random_examples_500_{}.dat".format(config.dataset, config.dataset)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    model_config = argparse.Namespace()
    model_config.d = config.d
    model_config.drop = config.drop
    # set seed
    set_seed(config.seed, cuda=config.use_cuda)
    # load dataset
    train = Interactions(dataset_path)  # all datasets
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T, test_random_path, config.Window)
    print(config)
    print(model_config)
    logging.info(config)
    logging.info(model_config)
    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        kld1=config.kld1,
                        kld2=config.kld2,
                        kld3=config.kld3,
                        model_args=model_config,
                        use_cuda=config.use_cuda)

    model.fit(train, verbose=True)

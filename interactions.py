import numpy as np
import pickle
import scipy.sparse as sp
from datetime import datetime

class Interactions(object):


    def __init__(self, file_path,
                 user_map=None,
                 item_map=None):
        user_ids = list()
        item_ids = list()
        time_ids = list()
        category_ids = list()
        item_cat_dict = dict()
        max_cat = 0
        # read users and items from file
        with open(file_path, 'r') as fin:
            for line in fin:
                u, i, c, timestamp = line.strip().split(' ')
                u = int(u) ## begin with 1
                i = int(i) ## begin with 1
                user_ids.append(u)
                item_ids.append(i)
                current_cats = []
                c = c.split(',') ##begin with 0
                date = datetime.fromtimestamp(int(timestamp))
                current_hour = date.hour
                time_ids.append(current_hour)
                for cat in c:
                    current_cats.append(int(cat) + 1)## mention that category begins with 0
                if max_cat<= max(current_cats):
                    max_cat = max(current_cats)
                category_ids.append(current_cats)

        user_ids = np.array([u - 1 for u in user_ids]) # begin with 0
        item_ids = np.array([i for i in item_ids]) # begin with 1
        time_ids = np.array(time_ids)

        self.num_users = len(set(user_ids))
        self.num_items = len(set(item_ids)) + 1

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.time_ids = time_ids

        self.category_ids = category_ids #begin with 1
        self.item_cat_dict = item_cat_dict
        self.num_categories = max_cat + 1
        self.users_test_random_dict = dict()


        self.sequences = None
        self.test_sequences = None
        self.valid_sequences = None



    def __len__(self):
        return len(self.user_ids)

    def to_sequence(self, sequence_length=5, target_length=1, test_random_path = None, Window = None):
        with open(test_random_path, "rb") as i:
            users_test_random_dict_source = pickle.load(i)
        for uid_str in users_test_random_dict_source:
            uid_current = uid_str - 1
            self.users_test_random_dict[uid_current] = []
            for random_item_str in users_test_random_dict_source[uid_str][:]:
                self.users_test_random_dict[uid_current].append(random_item_str)

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = np.expand_dims(self.item_ids[sort_indices], axis=1)
        category_ids = np.array(self.category_ids)
        category_ids = category_ids[sort_indices]
        time_ids = np.expand_dims(self.time_ids[sort_indices], axis=1)####
        item_ids = np.concatenate((item_ids, time_ids, category_ids), axis=1)

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        counts = np.array([Window if c >= Window else c for c in counts])

        counts = counts - 2
        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])

        num_features = 3
        sequences = np.zeros((num_subsequences, sequence_length, num_features),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length, num_features),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)
        test_sequences = np.zeros((self.num_users, sequence_length, num_features),
                                  dtype=np.int64)
        valid_sequences = np.zeros((self.num_users, sequence_length, num_features),
                                  dtype=np.int64)
        test_target = np.zeros((self.num_users, 1, num_features),
                                  dtype=np.int64)
        valid_target = np.zeros((self.num_users, 1, num_features),
                               dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)
        valid_users = np.empty(self.num_users,dtype=np.int64)
        _uid = None
        flag = 0
        i = 0
        for m, (uid,
                item_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           indices,
                                                           max_sequence_length, Window)):
            if uid != _uid:
                flag = 1
                test_sequences[uid][:] = item_seq[ - sequence_length - 1: - 1]
                test_target[uid][:] = item_seq[- 1:]
                test_users[uid] = uid
                _uid = uid
                continue
            if flag == 1:
                flag = 0
                valid_sequences[uid][:] = item_seq[- sequence_length - 1: - 1]
                valid_target[uid][:] = item_seq[- 1:]
                valid_users[uid] = uid
                continue

            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[: sequence_length]
            sequence_users[i] = uid
            i += 1
        self.sequences = SequenceInteractions(sequence_users, sequences, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences, test_target)
        self.valid_sequences = SequenceInteractions(valid_users, valid_sequences, valid_target)

class SequenceInteractions(object):
    def __init__(self,
                 user_ids,
                 sequences,
                 targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]
def _sliding_window(tensor, window_size, Window = 200, step_size=1):
    if len(tensor)>Window:
        tensor = tensor[-Window:]
    if len(tensor) - window_size >= 2:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor) + 2
        tensor = np.pad(tensor, ((num_paddings, 0), (0, 0)), 'constant')
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length, Window):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]##cut off valid and test
        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length, Window):
            yield (user_ids[i], seq)

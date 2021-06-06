import numpy as np
from utils import *

def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):

    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def evaluate_ranking(model, train=None, k=10, is_valid = False):

    if is_valid == False:
        test_data = train.test_sequences
    elif is_valid == True:
        test_data = train.valid_sequences
    NDCG = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    HT = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    Precision = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    Recall = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    AUC = np.array([0.0])

    user_ids = test_data.user_ids
    targets = test_data.targets
    for (i,(batch_users, batch_targets)) in enumerate(minibatch( user_ids, targets, batch_size = 512)):
        batch_test_items = []
        for index, current in enumerate(zip(batch_users, batch_targets)):
            (user_id,target) = current
            target = target[0][0]
            test_items = []
            test_items.extend(train.users_test_random_dict[user_id])
            test_items.append(target)

            test_items = np.array(test_items, dtype=int).reshape(-1, 1)
            batch_test_items.append(test_items[np.newaxis:])

        test_items = np.stack(batch_test_items, axis = 0)
        predictions = -model.predict(batch_users, valid_flag = is_valid, item_ids = test_items)

        for prediction in predictions:
            prediction_1 = -prediction.argsort()
            rank = prediction_1.argsort()[0]
            AUC += (1 - rank / (1 * 500))
            if rank < 1:
                NDCG[0] += 1 / np.log2(rank + 2)
                HT[0] += 1
                Precision[0] += 1 / 1
                Recall[0] += 1
            if rank < 5:
                NDCG[1] += 1 / np.log2(rank + 2)
                HT[1] += 1
                Precision[1] += 1 / 5
                Recall[1] += 1
            if rank < 10:
                NDCG[2] += 1 / np.log2(rank + 2)
                HT[2] += 1
                Precision[2] += 1 / 10
                Recall[2] += 1
            if rank < 15:
                NDCG[3] += 1 / np.log2(rank + 2)
                HT[3] += 1
                Precision[3] += 1 / 15
                Recall[3] += 1
            if rank < 20:
                NDCG[4] += 1 / np.log2(rank + 2)
                HT[4] += 1
                Precision[4] += 1 / 20
                Recall[4] += 1
    NDCG = NDCG/len(user_ids)
    HT = HT / len(user_ids)
    Precision = Precision / len(user_ids)
    Recall = Recall / len(user_ids)
    AUC = AUC / len(user_ids)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    return NDCG, HT, F1, AUC


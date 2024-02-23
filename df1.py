from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import random

from sentence_transformers import SentenceTransformer
from tte_depth import StatDepth


def _depth_scores_y(x, y, model_card):
    model = SentenceTransformer(model_card)

    # encode all texts using the sentence transformer model
    F = model.encode(x)
    G = model.encode(y)

    # depth_rank_test returns depth scores for each corpus, along with a Q estimate, W test statistic from the Wilcoxon Rank Sum Test, and an associated p-value
    d = StatDepth()
    _, depth_scores_G, _, _, _ = d.depth_rank_test(F, G)

    # return 
    print(f"Depth: {depth_scores_G}")
    return depth_scores_G

def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _create_sample_weights(x, y, train_n, model_card):
    train_x = random.choices(x, k=train_n)
    depth_y = _depth_scores_y(train_x, y, model_card=model_card)
    sample_weights = _softmax(depth_y * -1)
    print(f"Softmaxed: {sample_weights}")
    return sample_weights

def df1(y_true, y_pred, x, y, average="micro", train_n = 500, model_card = 'all-MiniLM-L6-v2'):
    sample_weights = _create_sample_weights(x,y, train_n, model_card)
    _, _, f_score, _ = precision_recall_fscore_support(y_true, y_pred, sample_weight=sample_weights, average=average)
    return f_score
import numpy as np
from sklearn import metrics


def cal_score(y_true, y_score, threshold=0.5, topk=8):
    '''
    https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
    '''

    # check if each label has at least one sample
    idx = y_true.sum(0) > 0
    y_true = y_true[:, idx]
    y_score = y_score[:, idx]

    # cal scores
    score = {}
    score['roc_auc'] = metrics.roc_auc_score(y_true, y_score)
    score['converage_error'] = metrics.coverage_error(y_true, y_score)
    score[
        'rank_average_precision'] = metrics.label_ranking_average_precision_score(
            y_true, y_score)
    score['rank'] = metrics.label_ranking_loss(y_true, y_score)

    # binarization using threshold
    y_pred = np.zeros_like(y_score)
    y_pred[y_score > threshold] = 1
    score['zero_one'] = metrics.zero_one_loss(y_true, y_pred)
    cf_mat = metrics.multilabel_confusion_matrix(y_true, y_pred)
    score['confusion_matrix'] = cf_mat.tolist()

    # binarization using top-k logits
    y_pred = np.zeros_like(y_score)
    ind = np.argsort(-y_score, axis=1)[:, :topk]
    for i in range(y_pred.shape[0]):
        y_pred[i, ind[i, :]] = 1
    p, r, f, _ = metrics.precision_recall_fscore_support(y_true,
                                                         y_pred,
                                                         average='micro')
    score['precision'] = p
    score['recall'] = r
    score['f1'] = f

    return score

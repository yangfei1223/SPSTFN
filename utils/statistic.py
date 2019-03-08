# -*- coding:utf-8 -*-
from helper import evalExp, pxEval_maximizeFMeasure, getGroundTruth
import numpy as np
li_property = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp', 'FPR_wp', 'FNR_wp']


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def eval_road(li_pred, li_gt):
    print "Starting evaluation ..."
    thresh = np.array(range(0, 256)) / 255.0    # recall thresh
    # init result
    totalFP = np.zeros(thresh.shape)
    totalFN = np.zeros(thresh.shape)
    totalPosNum = 0
    totalNegNum = 0

    for i in range(len(li_pred)):
        pred = li_pred[i]
        # gt = li_gt[i] > 0
        gt = li_gt[i] == 1
        validArea = li_gt[i] < 250
        FN, FP, posNum, negNum = evalExp(gt, pred, thresh, validMap=None, validArea=validArea)

        assert FN.max() <= posNum, 'BUG @ poitive samples'
        assert FP.max() <= negNum, 'BUG @ negative samples'

        # collect results for whole category
        totalFP += FP
        totalFN += FN
        totalPosNum += posNum
        totalNegNum += negNum

    # if category_ok:
    print "Computing evaluation scores..."
    # Compute eval scores!
    eval_dict = pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh=thresh)

    for property in li_property:
        print '%s: %4.2f ' % (property, eval_dict[property] * 100,)

    print "Finished evaluating!"
    return eval_dict['MaxF']

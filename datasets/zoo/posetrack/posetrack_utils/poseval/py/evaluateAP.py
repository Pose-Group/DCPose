import numpy as np
import json
import os
import sys

from .eval_helpers import assignGTmulti, computeRPC, Joint, VOCap

def computeMetrics(scoresAll, labelsAll, nGTall):
    apAll = np.zeros((nGTall.shape[0] + 1, 1))
    recAll = np.zeros((nGTall.shape[0] + 1, 1))
    preAll = np.zeros((nGTall.shape[0] + 1, 1))
    # iterate over joints
    for j in range(nGTall.shape[0]):
        scores = np.zeros([0, 0], dtype=np.float32)
        labels = np.zeros([0, 0], dtype=np.int8)
        # iterate over images
        for imgidx in range(nGTall.shape[1]):
            scores = np.append(scores, scoresAll[j][imgidx])
            labels = np.append(labels, labelsAll[j][imgidx])
        # compute recall/precision values
        nGT = sum(nGTall[j, :])
        precision, recall, scoresSortedIdxs = computeRPC(scores, labels, nGT)
        if (len(precision) > 0):
            apAll[j] = VOCap(recall, precision) * 100
            preAll[j] = precision[len(precision) - 1] * 100
            recAll[j] = recall[len(recall) - 1] * 100
    idxs = np.argwhere(~np.isnan(apAll[:nGTall.shape[0],0]))
    apAll[nGTall.shape[0]] = apAll[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(recAll[:nGTall.shape[0],0]))
    recAll[nGTall.shape[0]] = recAll[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(preAll[:nGTall.shape[0],0]))
    preAll[nGTall.shape[0]] = preAll[idxs, 0].mean()

    return apAll, preAll, recAll


def evaluateAP(gtFramesAll, prFramesAll):

    distThresh = 0.5

    # assign predicted poses to GT poses
    scoresAll, labelsAll, nGTall, _ = assignGTmulti(gtFramesAll, prFramesAll, distThresh)

    # compute average precision (AP), precision and recall per part
    apAll, preAll, recAll = computeMetrics(scoresAll, labelsAll, nGTall)

    return apAll, preAll, recAll

#def evaluateAP(gtFramesAll, prFramesAll, outputDir, bSaveAll=True, bSaveSeq=False):
#
#    distThresh = 0.5
#
#    seqidxs = []
#    for imgidx in range(len(gtFramesAll)):
#        seqidxs += [gtFramesAll[imgidx]["seq_id"]]
#    seqidxs = np.array(seqidxs)
#
#    seqidxsUniq = np.unique(seqidxs)
#    nSeq = len(seqidxsUniq)
#
#    names = Joint().name
#    names['15'] = 'total'
#
#    if (bSaveSeq):
#        for si in range(nSeq):
#            print("seqidx: %d/%d" % (si+1,nSeq))
#
#            # extract frames IDs for the sequence
#            imgidxs = np.argwhere(seqidxs == seqidxsUniq[si])
#            seqName = gtFramesAll[imgidxs[0,0]]["seq_name"]
#
#            gtFrames = [gtFramesAll[imgidx] for imgidx in imgidxs.flatten().tolist()]
#            prFrames = [prFramesAll[imgidx] for imgidx in imgidxs.flatten().tolist()]
#
#            # assign predicted poses to GT poses
#            scores, labels, nGT, _ = eval_helpers.assignGTmulti(gtFrames, prFrames, distThresh)
#
#            # compute average precision (AP), precision and recall per part
#            ap, pre, rec = computeMetrics(scores, labels, nGT)
#            metricsSeq = {'ap': ap.flatten().tolist(), 'pre': pre.flatten().tolist(), 'rec': rec.flatten().tolist(), 'names': names}
#
#            filename = outputDir + '/' + seqName + '_AP_metrics.json'
#            print('saving results to', filename)
#            eval_helpers.writeJson(metricsSeq,filename)
#
#    # assign predicted poses to GT poses
#    scoresAll, labelsAll, nGTall, _ = eval_helpers.assignGTmulti(gtFramesAll, prFramesAll, distThresh)
#
#    # compute average precision (AP), precision and recall per part
#    apAll, preAll, recAll = computeMetrics(scoresAll, labelsAll, nGTall)
#    if (bSaveAll):
#        metrics = {'ap': apAll.flatten().tolist(), 'pre': preAll.flatten().tolist(), 'rec': recAll.flatten().tolist(),  'names': names}
#        filename = outputDir + '/total_AP_metrics.json'
#        print('saving results to', filename)
#        eval_helpers.writeJson(metrics,filename)
#
#    return apAll, preAll, recAll

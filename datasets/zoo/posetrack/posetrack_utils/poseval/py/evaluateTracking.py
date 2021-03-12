import numpy as np
import json
import os
import sys

#import poseval.py.eval_helpers as eval_helpers
#from poseval.py.eval_helpers import Joint
#import poseval.py.motmetrics as mm

import motmetrics as mm
from .eval_helpers import assignGTmulti, Joint, writeJson

def computeMetrics(gtFramesAll, motAll, outputDir, bSaveAll, bSaveSeq):

    assert(len(gtFramesAll) == len(motAll))

    nJoints = Joint().count
    seqidxs = []
    for imgidx in range(len(gtFramesAll)):
        seqidxs += [gtFramesAll[imgidx]["seq_id"]]
    seqidxs = np.array(seqidxs)

    seqidxsUniq = np.unique(seqidxs)

    # intermediate metrics
    metricsMidNames = ['num_misses', 'num_switches', 'num_false_positives', 'num_objects', 'num_detections']

    # final metrics computed from intermediate metrics
    metricsFinNames = ['mota','motp','pre','rec']

    # initialize intermediate metrics
    metricsMidAll = {}
    for name in metricsMidNames:
        metricsMidAll[name] = np.zeros([1,nJoints])
    metricsMidAll['sumD'] = np.zeros([1,nJoints])

    # initialize final metrics
    metricsFinAll = {}
    for name in metricsFinNames:
        metricsFinAll[name] = np.zeros([1,nJoints+1])

    # create metrics
    mh = mm.metrics.create()

    imgidxfirst = 0
    # iterate over tracking sequences
    # seqidxsUniq = seqidxsUniq[:20]
    nSeq = len(seqidxsUniq)

    # initialize per-sequence metrics
    metricsSeqAll = {}
    for si in range(nSeq):
        metricsSeqAll[si] = {}
        for name in metricsFinNames:
            metricsSeqAll[si][name] = np.zeros([1,nJoints+1])

    names = Joint().name
    names['15'] = 'total'

    for si in range(nSeq):
    #for si in range(5):
        print("seqidx: %d/%d" % (si+1,nSeq))

        # init per-joint metrics accumulator
        accAll = {}
        for i in range(nJoints):
            accAll[i] = mm.MOTAccumulator(auto_id=True)

        # extract frames IDs for the sequence
        imgidxs = np.argwhere(seqidxs == seqidxsUniq[si])
        imgidxs = imgidxs[:-1].copy()
        seqName = gtFramesAll[imgidxs[0,0]]["seq_name"]
        print(seqName)
        # create an accumulator that will be updated during each frame
        # iterate over frames
        for j in range(len(imgidxs)):
            imgidx = imgidxs[j,0]
            # iterate over joints
            for i in range(nJoints):
                # GT tracking ID
                trackidxGT = motAll[imgidx][i]["trackidxGT"]
                # prediction tracking ID
                trackidxPr = motAll[imgidx][i]["trackidxPr"]
                # distance GT <-> pred part to compute MOT metrics
                # 'NaN' means force no match
                dist = motAll[imgidx][i]["dist"]
                # Call update once per frame
                accAll[i].update(
                    trackidxGT,                 # Ground truth objects in this frame
                    trackidxPr,                 # Detector hypotheses in this frame
                    dist                        # Distances from objects to hypotheses
                )

        # compute intermediate metrics per joint per sequence
        for i in range(nJoints):
            metricsMid = mh.compute(accAll[i], metrics=metricsMidNames, return_dataframe=False, name='acc')
            for name in metricsMidNames:
                metricsMidAll[name][0,i] += metricsMid[name]
            s = accAll[i].events['D'].sum()
            if (np.isnan(s)):
                s = 0
            metricsMidAll['sumD'][0,i] += s

#        if (bSaveSeq):
        if False:
            # compute metrics per joint per sequence
            for i in range(nJoints):
                metricsMid = mh.compute(accAll[i], metrics=metricsMidNames, return_dataframe=False, name='acc')

                # compute final metrics per sequence
                if (metricsMid['num_objects'] > 0):
                    numObj = metricsMid['num_objects']
                else:
                    numObj = np.nan
                numFP = metricsMid['num_false_positives']
                metricsSeqAll[si]['mota'][0,i] = 100*(1. - 1.*(metricsMid['num_misses'] +
                                                    metricsMid['num_switches'] +
                                                    numFP) /
                                                    numObj)
                numDet = metricsMid['num_detections']
                s = accAll[i].events['D'].sum()
                if (numDet == 0 or np.isnan(s)):
                    metricsSeqAll[si]['motp'][0,i] = 0.0
                else:
                    metricsSeqAll[si]['motp'][0,i] = 100*(1. - (1.*s / numDet))
                if (numFP+numDet > 0):
                    totalDet = numFP+numDet
                else:
                    totalDet = np.nan
                metricsSeqAll[si]['pre'][0,i]  = 100*(1.*numDet /
                                                totalDet)
                metricsSeqAll[si]['rec'][0,i]  = 100*(1.*numDet /
                                        numObj)

            # average metrics over all joints per sequence
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['mota'][0,:nJoints]))
            metricsSeqAll[si]['mota'][0,nJoints] = metricsSeqAll[si]['mota'][0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['motp'][0,:nJoints]))
            metricsSeqAll[si]['motp'][0,nJoints] = metricsSeqAll[si]['motp'][0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['pre'][0,:nJoints]))
            metricsSeqAll[si]['pre'][0,nJoints]  = metricsSeqAll[si]['pre'] [0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['rec'][0,:nJoints]))
            metricsSeqAll[si]['rec'][0,nJoints]  = metricsSeqAll[si]['rec'] [0,idxs].mean()

            metricsSeq = metricsSeqAll[si].copy()
            metricsSeq['mota'] = metricsSeq['mota'].flatten().tolist()
            metricsSeq['motp'] = metricsSeq['motp'].flatten().tolist()
            metricsSeq['pre'] = metricsSeq['pre'].flatten().tolist()
            metricsSeq['rec'] = metricsSeq['rec'].flatten().tolist()
            metricsSeq['names'] = names

            filename = outputDir + '/' + seqName + '_MOT_metrics.json'
            print('saving results to', filename)
            #eval_helpers.writeJson(metricsSeq,filename)
            writeJson(metricsSeq,filename)

    # compute final metrics per joint for all sequences
    for i in range(nJoints):
        if (metricsMidAll['num_objects'][0,i] > 0):
            numObj = metricsMidAll['num_objects'][0,i]
        else:
            numObj = np.nan
        numFP = metricsMidAll['num_false_positives'][0,i]
        metricsFinAll['mota'][0,i] = 100*(1. - (metricsMidAll['num_misses'][0,i] +
                                                metricsMidAll['num_switches'][0,i] +
                                                numFP) /
                                                numObj)
        numDet = metricsMidAll['num_detections'][0,i]
        s = metricsMidAll['sumD'][0,i]
        if (numDet == 0 or np.isnan(s)):
            metricsFinAll['motp'][0,i] = 0.0
        else:
            metricsFinAll['motp'][0,i] = 100*(1. - (s / numDet))
        if (numFP+numDet > 0):
            totalDet = numFP+numDet
        else:
            totalDet = np.nan

        metricsFinAll['pre'][0,i]  = 100*(1.*numDet /
                                       totalDet)
        metricsFinAll['rec'][0,i]  = 100*(1.*numDet /
                                       numObj)

    # average metrics over all joints over all sequences
    idxs = np.argwhere(~np.isnan(metricsFinAll['mota'][0,:nJoints]))
    metricsFinAll['mota'][0,nJoints] = metricsFinAll['mota'][0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['motp'][0,:nJoints]))
    metricsFinAll['motp'][0,nJoints] = metricsFinAll['motp'][0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['pre'][0,:nJoints]))
    metricsFinAll['pre'][0,nJoints]  = metricsFinAll['pre'] [0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['rec'][0,:nJoints]))
    metricsFinAll['rec'][0,nJoints]  = metricsFinAll['rec'] [0,idxs].mean()

#    if (bSaveAll):
    if False:
        metricsFin = metricsFinAll.copy()
        metricsFin['mota'] = metricsFin['mota'].flatten().tolist()
        metricsFin['motp'] = metricsFin['motp'].flatten().tolist()
        metricsFin['pre'] = metricsFin['pre'].flatten().tolist()
        metricsFin['rec'] = metricsFin['rec'].flatten().tolist()
        metricsFin['names'] = names

        filename = outputDir + '/total_MOT_metrics.json'
        print('saving results to', filename)
#        eval_helpers.writeJson(metricsFin,filename)
        writeJson(metricsFin,filename)

    return metricsFinAll


def evaluateTracking(gtFramesAll, prFramesAll, outputDir, saveAll=True, saveSeq=False):

    distThresh = 0.5
    # assign predicted poses to GT poses
#    _, _, _, motAll = eval_helpers.assignGTmulti(gtFramesAll, prFramesAll, distThresh)
    _, _, _, motAll = assignGTmulti(gtFramesAll, prFramesAll, distThresh)


    # compute MOT metrics per part
    metricsAll = computeMetrics(gtFramesAll, motAll, outputDir, saveAll, saveSeq)

    return metricsAll

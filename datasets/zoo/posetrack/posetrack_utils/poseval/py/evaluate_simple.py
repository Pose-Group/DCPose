from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
from .evaluateAP import evaluateAP
from .evaluateTracking import evaluateTracking
from .eval_helpers import Joint, printTable, load_data_dir, getCum


def evaluate(gtdir, preddir, eval_pose=True, eval_track=True,
             eval_upper_bound=False):
    logger = logging.getLogger(__name__)
    gtFramesAll, prFramesAll = load_data_dir(['', gtdir, preddir])

    logger.info('# gt frames  : {}'.format(str(len(gtFramesAll))))
    logger.info('# pred frames: {}'.format(str(len(prFramesAll))))

    apAll = np.full((Joint().count + 1, 1), np.nan)
    preAll = np.full((Joint().count + 1, 1), np.nan)
    recAll = np.full((Joint().count + 1, 1), np.nan)
    cum = None
    track_cum = None
    if eval_pose:
        apAll, preAll, recAll = evaluateAP(gtFramesAll, prFramesAll)

    logger.info('Average Precision (AP) metric:')
    # printTable(apAll)
    cum = printTable(apAll)

    metrics = np.full((Joint().count + 4, 1), np.nan)
    # print(eval_track)
    if eval_track:
        # print(xy)
        metricsAll = evaluateTracking(
            gtFramesAll, prFramesAll, eval_upper_bound)

        for i in range(Joint().count + 1):
            metrics[i, 0] = metricsAll['mota'][0, i]
        metrics[Joint().count + 1, 0] = metricsAll['motp'][0, Joint().count]
        metrics[Joint().count + 2, 0] = metricsAll['pre'][0, Joint().count]
        metrics[Joint().count + 3, 0] = metricsAll['rec'][0, Joint().count]
        logger.info('Multiple Object Tracking (MOT) mmetrics:')
        # print('Multiple Object Tracking (MOT) mmetrics:')
        track_cum = printTable(metrics, motHeader=True)
    # return (apAll, preAll, recAll), metrics
    # print(xy)
    return cum, track_cum

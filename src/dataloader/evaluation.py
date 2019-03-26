# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

from collections import defaultdict

import itertools
import numpy as np
import skimage.morphology
from easydict import EasyDict as edict
from prettytable import PrettyTable

import measures
from misc import log
from misc.config import cfg
from misc.parallel import Parallel, delayed

_db_measures = {
        'J': measures.db_eval_iou,
        'F': measures.db_eval_boundary,
        'T': measures.db_eval_boundary,
        }

def db_eval_sequence(segmentations,annotations,measure='J',n_jobs=cfg.N_JOBS):

  """
  Evaluate video sequence results.
	Arguments:
		segmentations (list of ndarrya): segmentations masks.
		annotations   (list of ndarrya): ground-truth  masks.
    measure       (char): evaluation metric (J,F,T)
    n_jobs        (int) : number of CPU cores.
  Returns:
    results (list): ['raw'] per-frame, per-object sequence results.
  """

  results = {'raw':[]}
  for obj_id in annotations.iter_objects_id():
    results['raw'].append(Parallel(n_jobs=n_jobs)(delayed(_db_measures[measure])(
      an==obj_id,sg==obj_id) for an,sg in zip(annotations[1:-1],segmentations[1:-1])))

  for stat,stat_fuc in measures._statistics.iteritems():
    results[stat] = [float(stat_fuc(r)) for r in results['raw']]

  # Convert to 'float' to save correctly in yaml format
  for r in range(len(results['raw'])):
    results['raw'][r] = [float(v) for v in results['raw'][r]]

  results['raw'] = [[np.nan]+r+[np.nan] for r in results['raw']]

  return results

def db_eval(db,segmentations,measures,n_jobs=cfg.N_JOBS,verbose=True):

  """
  Evaluate video sequence results.
	Arguments:
		segmentations (list of ndarrya): segmentations masks.
		annotations   (list of ndarrya): ground-truth  masks.
    measure       (char): evaluation metric (J,F,T)
    n_jobs        (int) : number of CPU cores.
  Returns:
    results (dict): [sequence]: per-frame sequence results.
                    [dataset] : aggreated dataset results.
  """

  s_eval = defaultdict(dict)  # sequence evaluation
  d_eval = defaultdict(dict)  # dataset  evaluation

  for measure in measures:
    log.info("Evaluating measure: {}".format(measure))
    for sid in range(len(db)):
      sg = segmentations[sid]
      s_eval[sg.name][measure] = db_eval_sequence(sg,
          db[sg.name].annotations,measure=measure,n_jobs=n_jobs)

    for statistic in cfg.EVAL.STATISTICS:
      raw_data = np.hstack([s_eval[sequence][measure][statistic] for sequence in
        s_eval.keys()])
      d_eval[measure][statistic] = float(np.mean(raw_data))

  g_eval = {'sequence':dict(s_eval),'dataset':dict(d_eval)}

  return g_eval

def print_results(evaluation,method_name="-"):
  """Print result in a table"""

  metrics = evaluation['dataset'].keys()

  # Print results
  table = PrettyTable(['Method']+[p[0]+'_'+p[1] for p in
    itertools.product(metrics,cfg.EVAL.STATISTICS)])

  table.add_row([method_name]+["%.3f"%np.round(
    evaluation['dataset'][metric][statistic],3) for metric,statistic in
    itertools.product(metrics,cfg.EVAL.STATISTICS)])

  print("\n{}\n".format(str(table)))

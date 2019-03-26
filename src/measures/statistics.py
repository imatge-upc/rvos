import numpy as np
import warnings

def mean(X):
  """
  Compute average ignoring NaN values.
  """

  return np.nanmean(X)

def recall(X,threshold=0.5):
  """
  Fraction of values of X scoring higher than 'threshold'
  """
  return mean(np.array(X)>threshold)

def decay(X,n_bins=4):
  """
  Performance loss over time.
  """

  ids = np.round(np.linspace(1,len(X),n_bins+1)+1e-10)-1;
  ids = ids.astype(np.uint8)

  D_bins = [X[ids[i]:ids[i+1]+1] for i in range(0,4)]

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    D = np.nanmean(D_bins[0])-np.mean(D_bins[3])
  return D

def std(X):
  """
  Compute standard deviation.
  """
  return np.std(X)

_statistics = {
      'decay' : decay,
      'mean'  : mean,
      'recall': recall,
      'std'   : std
      }

def get(name):
  return _statistics[name]

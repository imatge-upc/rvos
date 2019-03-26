# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

import sys
import numpy as np
import scipy.spatial.distance as ssd
from tstab import *

def get_bijective_pairs(pairs,costmat):
	bij_pairs = bij_pairs_one_dim(pairs, costmat,0)
	bij_pairs = bij_pairs_one_dim(bij_pairs, costmat.T,1)
	return bij_pairs

def bij_pairs_one_dim(pairs, costmat, left_or_right):

	bij_pairs = []
	ids1      = np.unique(pairs[:,left_or_right])

	for ii in range(len(ids1)):
		curr_pairs = pairs[pairs[:,left_or_right]==ids1[ii],:].astype(np.int)
		curr_costs = costmat[curr_pairs[:,left_or_right], curr_pairs[:,1-left_or_right]]
		b = np.argmin(curr_costs)
		bij_pairs.append(curr_pairs[b])

	return np.array(bij_pairs)

def hist_cost_2(BH1,BH2):

	nsamp1,nbins=BH1.shape
	nsamp2,nbins=BH2.shape

	eps  = 2.2204e-16
	BH1n = BH1 / (np.sum(BH1,axis=1,keepdims=True)+eps)
	BH2n = BH2 / (np.sum(BH2,axis=1,keepdims=True)+eps)

	tmp1 = np.tile(np.transpose(np.atleast_3d(BH1n),[0,2,1]),(1,nsamp2,1))
	tmp2 = np.tile(np.transpose(np.atleast_3d(BH2n.T),[2,1,0]),(nsamp1,1,1))
	HC = 0.5*np.sum((tmp1-tmp2)**2/(tmp1+tmp2+eps),axis=2)

	return HC

def sc_compute(Bsamp,Tsamp,mean_dist,nbins_theta,nbins_r,r_inner,r_outer,out_vec):
	in_vec = (out_vec==0).ravel()
	nsamp = Bsamp.shape[1]
	r_array=ssd.squareform(ssd.pdist(Bsamp.T)).T

	theta_array_abs0=Bsamp[1,:].reshape(-1,1).dot(np.ones((1,nsamp))) - \
			np.ones((nsamp,1)).dot(Bsamp[1,:].reshape(1,-1))

	theta_array_abs1=Bsamp[0,:].reshape(-1,1).dot(np.ones((1,nsamp))) - \
			np.ones((nsamp,1)).dot(Bsamp[0,:].reshape(1,-1))

	theta_array_abs = np.arctan2(theta_array_abs0,theta_array_abs1).T
	theta_array=theta_array_abs-Tsamp.T.dot(np.ones((1,nsamp)))

	if mean_dist is None:
		mean_dist = np.mean(r_array[in_vec].T[in_vec].T)

	r_array_n = r_array / mean_dist

	r_bin_edges=np.logspace(np.log10(r_inner),np.log10(r_outer),nbins_r)
	r_array_q=np.zeros((nsamp,nsamp))

	for m in range(int(nbins_r)):
		r_array_q=r_array_q+(r_array_n<r_bin_edges[m])

	fz = r_array_q > 0
	theta_array_2 = np.fmod(np.fmod(theta_array,2*np.pi)+2*np.pi,2*np.pi)
	theta_array_q = 1+np.floor(theta_array_2/(2*np.pi/nbins_theta))

	nbins=nbins_theta*nbins_r
	BH=np.zeros((nsamp,nbins))
	count = 0
	for n in range(nsamp):
		fzn=fz[n]&in_vec
		Sn = np.zeros((nbins_theta,nbins_r))
		coords = np.hstack((theta_array_q[n,fzn].reshape(-1,1),
			r_array_q[n,fzn].astype(np.int).reshape(-1,1)))

		# SLOW...
		#for i,j in coords:
			#Sn[i-1,j-1] += 1

		# FASTER
		ids = np.ravel_multi_index((coords.T-1).astype(np.int),Sn.shape)
		Sn  = np.bincount(ids.ravel(),minlength = np.prod(Sn.shape)).reshape(Sn.shape)


		BH[n,:] = Sn.T[:].ravel()

	return BH.astype(np.int),mean_dist

def db_eval_t_stab(fgmask,ground_truth,timing=True):
	"""
	Calculates the temporal stability index between two masks
	Arguments:
					fgmask (ndarray):  Foreground Object mask at frame t
		ground_truth (ndarray):  Foreground Object mask at frame t+1
	Return:
							 T (ndarray):  Temporal (in-)stability
	   raw_results (ndarray):  Supplemental values
	"""

	cont_th = 3
	cont_th_up = 3

	# Shape context parameters
	r_inner     = 1.0/8.0
	r_outer     = 2.0
	nbins_r     = 5.0
	nbins_theta = 12.0

	poly1 = mask2poly(fgmask,cont_th)
	poly2 = mask2poly(ground_truth,cont_th)

	if len(poly1.contour_coords) == 0 or \
			len(poly2.contour_coords) == 0:
		return np.nan

	Cs1 = get_longest_cont(poly1.contour_coords)
	Cs2 = get_longest_cont(poly2.contour_coords)

	upCs1 = contour_upsample(Cs1,cont_th_up)
	upCs2 = contour_upsample(Cs2,cont_th_up)

	scs1,_=sc_compute(upCs1.T,np.zeros((1,upCs1.shape[0])),None,
			nbins_theta,nbins_r,r_inner,r_outer,np.zeros((1,upCs1.shape[0])))

	scs2,_=sc_compute(upCs2.T,np.zeros((1,upCs2.shape[0])),None,
			nbins_theta,nbins_r,r_inner,r_outer,np.zeros((1,upCs2.shape[0])))

	# Match with the 0-0 alignment
	costmat              = hist_cost_2(scs1,scs2)
	pairs ,max_sx,max_sy = match_dijkstra(np.ascontiguousarray(costmat))


	# Shift costmat
	costmat2 = np.roll(costmat ,-(max_sy+1),axis=1)
	costmat2 = np.roll(costmat2,-(max_sx+1),axis=0)

	# Redo again with the correct alignment
	pairs,_,_ = match_dijkstra(costmat2)

	# Put the pairs back to the original place
	pairs[:,0] = np.mod(pairs[:,0]+max_sx+1, costmat.shape[0])
	pairs[:,1] = np.mod(pairs[:,1]+max_sy+1, costmat.shape[1])

	pairs = get_bijective_pairs(pairs,costmat)

	pairs_cost = costmat[pairs[:,0], pairs[:,1]]
	min_cost   = np.average(pairs_cost)

	return min_cost
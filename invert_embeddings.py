#import mkl
#mkl.set_num_threads(2)

import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
import scipy as sp
import scipy.sparse, scipy.io, scipy.optimize
from scipy.special import expit
from scipy import sparse, stats
import torch
from torch import nn
from scipy.sparse import csgraph, linalg, csc_matrix
import random
import pickle as pk
import math
from scipy.sparse.csgraph import connected_components
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import argparse
import warnings
import sys
import networkx as nx
from collections import Counter

class Network:
	def __init__( self ):
		self.Adjacency = None
		self.Labels = None
		self.Embedding = None
		self.LR_Embedding = None
		self.G = None
		self.labeled = False
		self.rank = None

	def loadNetwork( self, network_filename, binarize=False ):
		"""
		Loads a network and its labels (if available)
		"""
		try:
			self.Adjacency = scipy.io.loadmat( network_filename )['network']
		except:
			self.Adjacency = scipy.io.loadmat( network_filename )['A']
		# destroy diag and binarize
		self.Adjacency.setdiag(0)
		if binarize:
			self.Adjacency.data = 1. * (self.Adjacency.data > 0)
    
		# Load labels - if available
		try:
			self.Labels = scipy.io.loadmat( network_filename )['group']
			try:
				self.Labels = self.Labels.todense().astype(np.int)
				self.Labels = np.array( self.Labels )
			except:
				self.Labels = self.Labels.astype(np.int)
			self.labeled = True
		except:
			self.Labels = [None]
			self.labeled = False
		#print("Labeled?", self.labeled)
		return

	def SBM( self, sizes, probs ):
		self.G = nx.stochastic_block_model( sizes, probs )
		self.Adjacency = nx.to_scipy_sparse_matrix(self.G)
		self.Labels = []
		[self.Labels.extend([i for _ in range(sizes[i])]) for i in range(len(sizes))]
		self.Labels = np.array( self.Labels )
		self.labeled = True
		return

	def standardize( self ):
		"""
		Make the graph undirected and select only the nodes
		belonging to the largest connected component.

		:param adj_matrix: sp.spmatrix
			Sparse adjacency matrix
		:param labels: array-like, shape [n]

		:return:
			standardized_adj_matrix: sp.spmatrix
			Standardized sparse adjacency matrix.
			standardized_labels: array-like, shape [?]
			Labels for the selected nodes.
		"""
		# copy the input
		standardized_adj_matrix = self.Adjacency.copy()

		# make the graph unweighted
		standardized_adj_matrix[standardized_adj_matrix != 0] = 1

		# make the graph undirected
		standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)

		# select the largest connected component
		_, components = connected_components(standardized_adj_matrix)
		c_ids, c_counts = np.unique(components, return_counts=True)
		id_max_component = c_ids[c_counts.argmax()]
		select = components == id_max_component
		standardized_adj_matrix = standardized_adj_matrix[select][:, select]
		if self.labeled:
			standardized_labels = self.Labels[select]
		else:
			standardized_labels = None

		# remove self-loops
		standardized_adj_matrix = standardized_adj_matrix.tolil()
		standardized_adj_matrix.setdiag(0)
		standardized_adj_matrix = standardized_adj_matrix.tocsr()
		standardized_adj_matrix.eliminate_zeros()

		self.Adjacency, self.Labels = standardized_adj_matrix, standardized_labels

		return
	def k_core( self, k ):
		"""
		Keeps the k-core of the input graph
		:param k: int 
		"""
		self.setNetworkXGraph()
		core_numbers = nx.core_number( self.G )
		select = [key for key,v in core_numbers.items() if v >=k]
		self.Adjacency = self.Adjacency[select][:, select]
		self.Labels =  self.Labels[select]
		
	def netmf_embedding(self, T, skip_max=False):
		"""
		Calculates the NetMF embedding for the network
		Parameters:
			rank (int): Low-rank approximation
			T (int): Optimization Window 
		""" 
		# Calculate embedding

		n = self.Adjacency.shape[0]
		lap, deg_sqrt = sp.sparse.csgraph.laplacian(self.Adjacency, normed=True, return_diag=True)
		lam, W = np.linalg.eigh((sp.sparse.identity(n) - lap).todense())
		perm = np.argsort(-np.abs(lam))
		lam, W = lam[perm], W[:,perm]
    
		deg_inv_sqrt_diag = sp.sparse.spdiags(1./deg_sqrt, 0, n, n)
		vol = self.Adjacency.sum()
		lam_trans = sp.sparse.spdiags(lam[1:] * (1-lam[1:]**T) / (1-lam[1:]), 0, n-1, n-1)
		if skip_max:
			self.Embedding =  np.log(1 + vol/T * deg_inv_sqrt_diag @ W[:,1:] @ lam_trans @ W[:,1:].T @ deg_inv_sqrt_diag)
		else:
			self.Embedding =  np.log(np.maximum(1., 1 + vol/T * deg_inv_sqrt_diag @ W[:,1:] @ lam_trans @ W[:,1:].T @ deg_inv_sqrt_diag))
		return

	def low_rank_embedding( self, rank ):
		# Low-rank approximation
		w, v = np.linalg.eigh( self.Embedding )
		order = np.argsort(np.abs(w))[::-1]
		w, v = w[order[:rank]], v[:,order[:rank]]
		self.LR_Embedding = v @ np.diag(w) @ v.T
		
		return

	def closenessCentrality( self ):
		cc = nx.closeness_centrality( self.G )
		return cc
	
	def pageRank( self ):
		pr_vector = nx.pagerank( self.G )
		return pr_vector

	def getAdjacency( self ):
		return self.Adjacency

	def get_LR_embedding( self ):
		return self.LR_Embedding

	def setNetworkXGraph( self ):
		self.G = nx.from_scipy_sparse_matrix(self.Adjacency)
		self.G.remove_edges_from(nx.selfloop_edges(self.G))
		
		return
		
	def getNetworkXGraph( self ):
		if not self.G:
			self.setNetworkXGraph()
		return self.G

	def getNodesVolume( self ):
		return self.Adjacency.shape[0], np.array(self.Adjacency.sum(axis=1)).flatten().sum()

	def getLabels( self ):
		return self.Labels
	
	def isLabeled( self ):
		return self.labeled
class Optimizer:
	def __init__(self, adjacency, embedding, filename, seq_number, rank, device=torch.device("cpu"), dtype=torch.double):
		self.n = adjacency.shape[0]
		self.device = device
		self.dtype = dtype
		self.pmi = torch.tensor(embedding, device=self.device, dtype=self.dtype, requires_grad=False)
		deg = np.array(adjacency.sum(axis=1)).flatten()
		self.vol = deg.sum()
		self.deg = torch.tensor(deg, device=self.device, dtype=self.dtype, requires_grad=False)
		self.filename = os.path.splitext( filename )[0]
		self.sequence = seq_number
		if not os.path.exists(self.filename + '_networks'):
   	 		os.makedirs(self.filename + '_networks')
		self.folder = self.filename + '_networks/'
		self.rank = rank
		self.shift = 0.
		self.adjacency = torch.tensor( adjacency.todense(), device=self.device, dtype=self.dtype, requires_grad=False)

	def learnNetwork( self, max_iter=50, method='autoshift' ):
		
		# FUNCTIONS
		def pmi_loss_10_elt_param(elts, n, logit_mode='raw', vol=0., skip_max=False, given_edges=False ):
			elts_tensor = torch.tensor(elts, device=self.device, dtype=self.dtype, requires_grad=True)
			adj_recon = torch.zeros(n,n, device=self.device, dtype=self.dtype)
			if logit_mode == 'individual':
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = torch.sigmoid(elts_tensor)
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=True, shift=0.)
			elif logit_mode == 'raw':
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = elts_tensor
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=False)
			elif logit_mode == 'softmax':
				adj_recon[np.triu_indices(n,1)] = torch.nn.functional.softmax(elts_tensor, dim=0) * (vol/2)
			elif logit_mode == 'autoshift':
				self.shift = 0.
				for i in range(10):
					self.shift = self.shift - (torch.sigmoid(elts_tensor+self.shift).sum() - (vol/2)) / (torch.sigmoid(elts_tensor+self.shift) * (1. - torch.sigmoid(elts_tensor+self.shift))).sum()
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = torch.sigmoid(elts_tensor+self.shift)
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=True, shift=shift)
        
			adj_recon = adj_recon + adj_recon.T
			deg_recon = adj_recon.sum(dim=0)
			vol_recon = deg_recon.sum()
			#with torch.no_grad():
			#	print( "Adjacency error: ", (adj_recon - self.adjacency).pow(2).sum() / (self.adjacency).pow(2).sum() )
			#	print( "Min Degree:", torch.min( deg_recon ) )
			#	print( "Max Degree:", torch.max( deg_recon ) )

			p_recon = (1. / deg_recon)[:,np.newaxis] * adj_recon
			p_recon_2 = p_recon @ p_recon
    
			p_recon_5 = (p_recon_2 @ p_recon_2) @ p_recon
			p_geo_series_recon = ( ((p_recon + p_recon_2) @ (torch.eye(n) + p_recon_2)) + p_recon_5 ) @ (torch.eye(n) + p_recon_5)
    			
			if skip_max:
				pmi_recon_exact = torch.log((vol_recon/10.) * p_geo_series_recon * (1. / deg_recon)[np.newaxis,:])
			else:
				pmi_recon_exact = torch.log(torch.clamp((vol_recon/10.) * p_geo_series_recon * (1. / deg_recon)[np.newaxis,:], min=1.))
			loss_pmi = (pmi_recon_exact - self.pmi).pow(2).sum() / (self.pmi).pow(2).sum()
			loss_deg = (deg_recon - self.deg).pow(2).sum() / self.deg.pow(2).sum()
			loss_vol = (vol_recon - self.vol).pow(2) / (self.vol**2)
    
			loss = loss_pmi
			print('{}. Loss: {}\t PMI: {}\t Vol: {}\t Deg: {:.2f}'.format(self.iter_num, math.sqrt( loss.item() ), math.sqrt( loss_pmi.item() ), loss_vol.item(), loss_deg.item()))
			loss.backward()
			with torch.no_grad():
				#if self.iter_num == 150:
				#	print("Loss: {}, Error: {}".format( loss.item(), (adj_recon - self.adjacency).pow(2).sum() / (self.adjacency).pow(2).sum() ) )
				if torch.isnan(loss):
					pass
					#print("Loss is nan on the following adj, p_sym, pmi:")
					#print(adj_recon)
					#print(p_recon)
					#print(pmi_recon_exact)
					#print(np.linalg.norm(pmi_recon_exact.detach().numpy() - pmi.detach().numpy()) / np.linalg.norm(pmi.detach().numpy()), 
				#      np.linalg.norm(pmi_recon_exact.detach().numpy() - pmi_exact) / np.linalg.norm(pmi_exact))
			gradients = elts_tensor.grad.numpy().flatten()
			#print("Nan in gradient?", np.argwhere( np.isnan(gradients) ) )
			#print("Gradient norm: ", np.linalg.norm( gradients ))
			return loss, gradients

		def callback_elt_param(x_i):
			self.elts = x_i
			self.iter_num += 1
			if self.iter_num % 5 == 0:
				np.save( 'adj_recon/' +  self.filename + '_' + self.rank +'_recon_elts.npy', expit(self.elts + self.shift.detach().numpy()))
		
		# MAIN OPTIMIZATION
		np.random.seed()
		self.elts = np.random.uniform(0,1, size=(self.n*self.n-self.n) // 2 )
		self.iter_num = 0
		self.elts *= 0
		res = scipy.optimize.minimize(pmi_loss_10_elt_param, x0=self.elts, 
                              args=(self.n,'autoshift',self.vol, False), jac=True, method='L-BFGS-B',
                             callback=callback_elt_param,
                              tol=np.finfo(float).eps, 
                                  options={'maxiter':max_iter, 'ftol':np.finfo(float).eps, 'gtol':np.finfo(float).eps}
                             )
	
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def main():
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
	
	# Read arguments
	parser = argparse.ArgumentParser(description='Define filename, window, rank and limit')
	parser.add_argument('-f', '--filename', required=True, help="Relative path to dataset file")
	parser.add_argument('-i', '--it', required=False, default="1", help="Number of iteration")
	parser.add_argument('-w', '--window', required=False, default="10", help="Window for SGD (default is 10)")
	parser.add_argument('-r', '--rank', required=False, default="128", help="Rank of approximation for embedding (default is 128)")
	parser.add_argument('-l', '--limit', required=False, default="4", help="Limit the number of threads (default is 4)")
	args = parser.parse_args()
	
	# Check validity of given arguments
	check_positive( args.it )
	window = check_positive( args.window )
	rank = check_positive( args.rank )
	
	# Limit number of threads
	#thread_limit( args.limit )
	
	# Network instance
	N = Network( )
	N.loadNetwork( args.filename, True )
	N.standardize()
	#N.k_core( 2 )
	#print( "Nodes: ", N.getAdjacency().shape )
	skip_max = False
	#if args.filename == "cora.mat" or args.filename == "citeseer.mat":
	#	skip_max = False 
	N.netmf_embedding( window, skip_max )
	N.low_rank_embedding( rank )
	
	# Learn Adjacency Matrix having same embedding
	P = Optimizer( N.getAdjacency(), N.get_LR_embedding(), args.filename, args.it, args.rank )
	P.learnNetwork() 

if __name__ == "__main__":
    main()

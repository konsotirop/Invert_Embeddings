import networkx as nx
import numpy as np
from scipy import sparse, io

sizes = [500,500]
probs = [[0.1,0.1],[0.01,0.01]]

G = nx.stochastic_block_model( sizes, probs )
S = nx.to_scipy_sparse_matrix( G )

labels = np.zeros((1000,2))
labels[:500,0] = 1
labels[500:,1] = 1

io.savemat('sbm1.mat', {'network':S, 'group':labels})       

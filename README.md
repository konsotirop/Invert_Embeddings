# Code for the paper:
# "DeepWalkings Backwards: From Embeddings Back to Graphs"


1. We include a DEMO file in the form of Jupyter Notebook (Demo.ipynb)
that goes through the main parts of our paper.  

2. A detailed description of the code and files used follows:  

File __"invert_embeddings.py"__ implements the optimization based procedure, that given as input
a low-dimensional embeddings of a network G, tries to learn a new network G' that has the same
embedding as G.  

Parameters:  

- __-f__: filename (networks are assumed to be in .mat format.

        Each .mat file contains two MATLAB sparse
        arrays:  
		   i. 'network' being the NXN sparse adjacency matrix,   
		   ii. 'group' being an NxC sparse matrix (C the number of labels), where each row
			i represents a node, and each column j is either 1 or 0, if node i has
			label j.
- __-w__: window-size (T). Default value is 10.
- __-r__: rank (The rank of the low-dimensional embedding of G that will be given as input). Default is 128.

File __"network_stats.py"__ takes as input the reconstructed networks from the above script and prints and plots
various graph properties for both the original and the reconstructed networks.
	
	Line 544 iterates over different ranks. Results should be first produced by running the above script (invert_embeddings.py).
	As for now, it only 'iterates' over rank 128, otherwise please change accordingly.  
Parameters:  
- __-f__: filename (the same as above)
- __-m__: binarization method (default is the one used in the paper, others include "add_edge" keeping the top-m edges). 

3. Usage example:  
	python invert_embeddings.py -f PPI.mat -r 128  
	python network_stats.py -f PPI.mat

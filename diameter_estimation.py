def getDiameter( Gc, approximate=True, seeds=100 ):
	import random
	import numpy as np
	if not approximate:
		diameter =  nx.diameter(Gc)
	else:
		src_nodes = random.sample(Gc.nodes(), seeds  )
		diameter = 0
		some_pairs = []
		for src in src_nodes:
			layers, distances = startBFS( Gc, src )
			diameter = max(diameter,max(layers.keys()))
			some_pairs.extend( distances ) # for effective diameter
	return diameter, np.percentile( some_pairs, 90 )

def startBFS( GC, src ):
	from collections import deque, defaultdict

	# Mark all the vertices as not visited 
	visited = dict.fromkeys(GC.nodes())
	# Initialize dictionary with nodes per layer
	layers = defaultdict( int )
	dist = []
	# Create a queue for BFS 
	queue = deque() 

	# Mark the source node as  
	# visited and enqueue it 
	queue.append((src,0)) 
	visited[src] = True

	while queue: 

		# Dequeue a vertex from  
		# queue and print it 
		u, l = queue.popleft() 
		layers[l] += 1
		dist.append( l )
		for v in GC.neighbors(u): 
			if not visited[v]:
				queue.append((v,l+1)) 
				visited[v] = True

	return layers, dist

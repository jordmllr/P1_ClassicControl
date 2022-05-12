# README
This project implements the MAP-Elites algorithm to evolve spiking neural networks to perform on classic reinforcement learning control tasks such as pole cart and mountain car and examine the network characteristics of the best performers. 

## MAP-Elites
MAP-Elites quantizes a phenotypic space and then keeps the best performer in each cell of the quantized space. This provides not only a way to optimize to a single solution, but also to explore a diverse set of options in the phenotypic space. In this project, the phenotypic space has 2 dimensions: clustering coefficient and average path length of the evolved networks.

## Archive
The archive is where phenotypic entries are stored. Each entry in the archive is the best-so-far performer in that slot on the quantized phenotypic space.

Entries are fully defined by 3 elements:
1. index - n-dim tuple indicating position in feature space
2. genotype - 2-d numpy array encoding adjacency matrix of network
3. performance - scalar value from evaluation of the genotype


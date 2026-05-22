import pickle
import networkx as nx
import random

print("Loading full Pune graph...")

with open("graph/graph_data.pkl", "rb") as f:
    G = pickle.load(f)

print("Total nodes in full graph:", len(G.nodes))

# sample nodes
sample_size = 2000
nodes = random.sample(list(G.nodes), sample_size)

print("Creating subgraph...")

subG = G.subgraph(nodes).copy()

print("Subgraph nodes:", len(subG.nodes))
print("Subgraph edges:", len(subG.edges))

with open("graph/subgraph.pkl", "wb") as f:
    pickle.dump(subG, f)

print("Subgraph saved successfully")
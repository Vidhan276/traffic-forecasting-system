import pickle
import osmnx as ox

print("Loading graph...")

with open("graph/graph_data.pkl", "rb") as f:
    G = pickle.load(f)

print("Displaying Pune road network...")

ox.plot_graph(G)
import osmnx as ox
import networkx as nx
import pickle

city = "Pune, India"

print("Downloading Pune road network...")

G = ox.graph_from_place(city, network_type="drive")

print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

# Save graph
with open("graph/graph_data.pkl", "wb") as f:
    pickle.dump(G, f)

print("Graph saved successfully")
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# === Costruzione grafo NetworkX ===
G = nx.Graph()

nodes = {
    1: {"type": "PMU"},
    2: {"type": "PMU"},
    3: {"type": "normal"},
    4: {"type": "normal"},
    5: {"type": "normal"},
    6: {"type": "normal"},
    7: {"type": "PMU"},
    8: {"type": "normal"},
    9: {"type": "normal"},
    10: {"type": "normal"},
    11: {"type": "normal"},
    12: {"type": "CC"},
}

for node_id, attr in nodes.items():
    G.add_node(node_id, **attr)

edges = [
    (1, 4, 1), (2, 4, 3), (2, 5, 4), (3, 5, 5), (3, 6, 1),
    (4, 11, 2), (5, 11, 7), (5, 6, 2), (6, 9, 2), (6, 8, 1),
    (6, 3, 1), (7, 8, 5), (8, 9, 2), (9, 10, 1), (10, 12, 8), (11, 12, 1),
]

for u, v, latency in edges:
    G.add_edge(u, v, latency=latency)

# === Feature dei nodi: codifica one-hot ===
type_encoding = {"PMU": [1, 0, 0], "CC": [0, 1, 0], "normal": [0, 0, 1]}

for node in G.nodes(data=True):
    node_type = node[1]["type"]
    G.nodes[node[0]]["x"] = type_encoding[node_type]

# === Conversione a PyG Data ===
data = from_networkx(G) # Questa riga converte il grafo networkx G in un oggetto torch_geometric.data.Data. data.edge_index: matrice 2×num_edges che rappresenta gli archi
data.x = torch.tensor([G.nodes[n]["x"] for n in G.nodes()], dtype=torch.float) # matrice num_nodes×3 che rappresenta le feature dei nodi
data.edge_attr = torch.tensor([G.edges[e]["latency"] for e in G.edges()], dtype=torch.float).view(-1, 1) # matrice num_edges×1 che rappresenta le feature degli archi (latenza) ( come fosse un vettore di latenze, una per arco )

# === Verifica ===
print("Feature dei nodi (x):", data.x.shape)             # [num_nodes, 3]
print("Edge index:", data.edge_index.shape)              # [2, num_edges]
print("Edge attr (latency):", data.edge_attr.shape)      # [num_edges, 1]

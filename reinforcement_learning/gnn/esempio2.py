import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

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

class GraphPolicyNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels): #in_channels: numero di feature per nodo (3, ovvero PMU, CC, normal), hidden_channels: numero di neuroni ( embedded nello strato nascosto)
        super(GraphPolicyNetwork, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels) # Primo strato convoluzionale, produce embedding di dimensione hidden_channels
        self.conv2 = GCNConv(hidden_channels, hidden_channels) # Secondo strato convoluzionale, raffina l'embedding
        self.output = nn.Linear(hidden_channels, 1) # Strato di output, ovvero produce un punteggio per ciascun nodo

    def forward(self, x, edge_index): # x: matrice num_nodes×3, edge_index: matrice 2×num_edges
        x = self.conv1(x, edge_index) # primo scambio convoluzionale
        x = F.relu(x) # rende non lineare l'output
        x = self.conv2(x, edge_index) # secondo scambio convoluzionale
        x = F.relu(x) # rende non lineare l'output
        logits = self.output(x).squeeze(-1) # logits: punteggio per ciascun nodo, e squeeze(-1) rimuove l'ultima dimensione [num_nodes, 1] → [num_nodes]
        probs = F.softmax(logits, dim=0) # softmax: trasforma i logit in probabilità normalizzate (sommate = 1), ogni probs[i] rappresenta la probabilità di piazzare un PDC sul nodo i
        return probs

# === INIZIALIZZAZIONE DEL MODELLO ===
model = GraphPolicyNetwork(in_channels=data.x.shape[1], hidden_channels=32)

# === ESECUZIONE DI UN FORWARD PASS ===
with torch.no_grad():
    action_probs = model(data.x, data.edge_index)
    print("Probabilità di selezionare ciascun nodo:")
    for i, p in enumerate(action_probs):
        print(f"Nodo {i+1}: {p.item():.4f}")
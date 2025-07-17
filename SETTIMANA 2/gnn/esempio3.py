import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# === ENVIRONMENT ===
class PDCEnv:
    def __init__(self, graph, pmus, cc):
        self.original_graph = graph
        self.pmus = pmus
        self.cc = cc
        self.reset()

    def reset(self):
        self.graph = self.original_graph.copy()
        self.pdcs = set()
        self.covered_pmus = set()
        self.done = False
        return {"pdcs": set(self.pdcs), "pmus": set(self.pmus), "cc": self.cc}

    def _check_validity(self):
        for pmu in self.pmus:
            if not any(neigh in self.pdcs for neigh in self.graph.neighbors(pmu)):
                return False
        for pdc in self.pdcs:
            if pdc not in self._get_first_last_pdcs():
                if not any(n in self.pdcs for n in self.graph.neighbors(pdc)):
                    return False
        if not any(self.cc in self.graph.neighbors(pdc) for pdc in self.pdcs):
            return False
        return True

    def _get_first_last_pdcs(self):
        first = {pdc for pdc in self.pdcs if any(n in self.pmus for n in self.graph.neighbors(pdc))}
        last = {pdc for pdc in self.pdcs if self.cc in self.graph.neighbors(pdc)}
        return first.union(last)

    def _compute_latency(self):
        total = 0
        for pmu in self.pmus:
            try:
                subgraph_nodes = list(self.pdcs.union({pmu, self.cc}))
                subgraph = self.graph.subgraph(subgraph_nodes)
                path = nx.shortest_path(subgraph, source=pmu, target=self.cc, weight="latency")
                for node in path:
                    if node != pmu and node != self.cc and node not in self.pdcs:
                        return float("inf")
                total += sum(self.graph[u][v]["latency"] for u, v in zip(path[:-1], path[1:]))
            except:
                return float("inf")
        return total

    def step(self, action):
        if action in self.pdcs or self.done or self.graph.nodes[action]["type"] in {"PMU", "CC"}:
            return self._get_state(), -10.0, self.done, {"info": "Invalid action"}
        
        self.pdcs.add(action)
        self.covered_pmus = {pmu for pmu in self.pmus if any(n in self.pdcs for n in self.graph.neighbors(pmu))}
        all_covered = len(self.covered_pmus) == len(self.pmus)
        valid = self._check_validity()
        latency = self._compute_latency() if valid else float('inf')
        
        # === NORMALIZZAZIONE ===
        MAX_LATENCY = 100.0  # valore empirico (puoi modificarlo)
        normalized_latency = latency / MAX_LATENCY if latency < float('inf') else 1e6

        penalty = len(self.pdcs) * 2.0  # aumentato per distinguere meglio
        bonus = 50 * len(self.covered_pmus)

        if valid and all_covered:
            reward = -normalized_latency - penalty
        elif valid:
            reward = -normalized_latency - penalty + bonus
        else:
            reward = -100.0

        self.done = valid and all_covered and latency < float('inf')

        # === DEBUG ===
        if self.done:
            print(f"[DEBUG] ✅ Episodio completato con successo!")
            print(f"[DEBUG]     Latency: {latency:.2f}")
            print(f"[DEBUG]     Normalized Latency: {normalized_latency:.4f}")
            print(f"[DEBUG]     Num PDCs: {len(self.pdcs)}")
            print(f"[DEBUG]     Reward finale: {reward:.2f}")
            print(f"[DEBUG]     PDCs scelti: {self.pdcs}")

        return self._get_state(), reward, self.done, {}


    def _get_state(self):
        return {"pdcs": set(self.pdcs), "pmus": set(self.pmus), "cc": self.cc}


# === GNN MODEL ===
class GraphPolicyNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        logits = self.out(x).squeeze(-1)
        temperature = 0.5  # o 0.1
        probs = F.softmax(logits / temperature, dim=0)  
        return probs


import matplotlib.pyplot as plt

def plot_solution_graph(G, pmus, cc, pdcs, paths):
    pos = nx.spring_layout(G, seed=42)  # per layout stabile
    plt.figure(figsize=(10, 8))

    # Disegna archi normali con etichetta di latenza
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")
    edge_labels = {(u, v): G[u][v]["latency"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Disegna path attivi (in evidenza)
    for path in paths:
        edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='blue', width=2)

    # Disegna nodi con colori diversi
    node_colors = []
    for n in G.nodes():
        if n in pmus:
            node_colors.append("skyblue")
        elif n in pdcs:
            node_colors.append("green")
        elif n == cc:
            node_colors.append("red")
        else:
            node_colors.append("lightgray")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Legenda personalizzata
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='PMU'),
        Patch(facecolor='green', edgecolor='black', label='PDC selezionato'),
        Patch(facecolor='red', edgecolor='black', label='Control Center (CC)'),
        Patch(facecolor='lightgray', edgecolor='black', label='Altro nodo'),
        Patch(color='blue', label='Path attivo'),
        Patch(color='gray', label='Arco normale (latency)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.title("Topologia della rete con i PDC selezionati e i path dai PMU al CC")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    
# === COSTRUZIONE GRAFO ===
G = nx.Graph()
nodes = {
    1: {"type": "PMU"}, 2: {"type": "PMU"}, 3: {"type": "normal"}, 4: {"type": "normal"},
    5: {"type": "normal"}, 6: {"type": "normal"}, 7: {"type": "PMU"}, 8: {"type": "normal"},
    9: {"type": "normal"}, 10: {"type": "normal"}, 11: {"type": "normal"}, 12: {"type": "CC"},
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

type_encoding = {"PMU": [1, 0, 0], "CC": [0, 1, 0], "normal": [0, 0, 1]}
for node in G.nodes(data=True):
    node[1]["x"] = type_encoding[node[1]["type"]]

data = from_networkx(G)
data.x = torch.tensor([G.nodes[n]["x"] for n in G.nodes()], dtype=torch.float)
data.edge_attr = torch.tensor([G.edges[e]["latency"] for e in G.edges()], dtype=torch.float).view(-1, 1)

# === TRAINING LOOP ===
env = PDCEnv(G, pmus=[1, 2, 7], cc=12)
model = GraphPolicyNetwork(in_channels=3, hidden_channels=16) # 3 feature per nodo (PMU, CC, normal), 16 neuroni nello strato nascosto, quindi ogni nodo ha 16 neuroni nell'embedding ( viene trasformato in un embedding di dimensione 16 )
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam è un tipo particolare di ottimizzatore, ovvero un algoritmo che aggiorna i pesi dei neuroni dopo ogni passo dell’allenamento per ridurre l’errore (cioè per migliorare il modello).
# === All'interno del training loop ===
episode_rewards = []

for episode in range(1000):
    state = env.reset() # stato iniziale dell'ambiente
    log_probs = [] # salva i log probabilità delle azioni scelte
    rewards = [] # salva i reward ottenuti
    valid_episode = True  # <-- default valido

    for t in range(10): # l'agente può fare al massimo 10 azioni in un episodio ( quindi posiziona al massimo 10 PDC )
        probs = model(data.x, data.edge_index) # calcola le probabilità delle azioni (ovvero dei nodi) da scegliere
        print(f"[DEBUG] Probs: {probs.detach().numpy()}")  # LOG dei logits trasformati

        mask = torch.ones_like(probs) # maschera per escludere PMU, CC e PDC già scelti
        for i, n in enumerate(G.nodes()):
            if n in state["pdcs"] or G.nodes[n]["type"] in {"PMU", "CC"}:
                mask[i] = 0.0

        masked_probs = probs * mask # applica la maschera alle probabilità
        total = masked_probs.sum() # somma delle probabilità restanti

        if total.item() <= 0.0 or torch.isnan(total): # Se le probabilità sono zero o non valide (NaN), forza una distribuzione uniforme tra le azioni valide. Ma quell’episodio sarà ignorato nel training.
            print(f"[DEBUG] Probabilità non valide: forzata distribuzione uniforme.")
            masked_probs = mask / mask.sum()
            valid_episode = False
        else:
            masked_probs = masked_probs / total

        m = torch.distributions.Categorical(masked_probs) # Crea una distribuzione discreta con le probabilità mascherate ( vettore di probabilità per ciascun nodo valido )
        action_index = m.sample() # campiona un indice di nodo in base alle probabilità mascherate ( come se lanciassi un dado con le probabilità dei nodi validi )
        action_node = list(G.nodes())[action_index.item()] # cerca il nodo corrispondente all'indice scelto

        state, reward, done, _ = env.step(action_node) # esegue l'azione nell'ambiente
        log_probs.append(m.log_prob(action_index)) # salva il logaritmo della probabilità dell'azione scelta
        rewards.append(reward) # salva il reward ottenuto

        if done:
            break

    total_reward = sum(rewards) # somma dei reward ottenuti nell'episodio

    # Clip reward inf/nan
    if not torch.isfinite(torch.tensor(total_reward)): # se il reward è inf o NaN, lo imposta a -10.0
        total_reward = -10.0
    episode_rewards.append(total_reward)

    # === WARM-UP: non fare backward nei primi episodi ===
    if episode < 10:
        continue

    if valid_episode and log_probs: # se l'episodio è valido e ci sono log probabilità
        total_reward_tensor = torch.tensor(total_reward, dtype=torch.float) # converte il reward totale in un tensore
        loss = -torch.stack(log_probs).mul(total_reward_tensor).sum() # calcola la loss come il negativo del prodotto tra i log probabilità e il reward totale -> se ho fatto buone azioni, la loss sarà bassa (perché il reward è positivo), altrimenti sarà alta (perché il reward è negativo)
        optimizer.zero_grad() # azzera i gradienti dell'ottimizzatore
        loss.backward() # calcola i gradienti, ogni peso w del tuo modello ha associato un valore chiamato w.grad
        optimizer.step() # aggiorna i pesi del modello

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

# === Mostra grafo finale ===
final_paths = []
for pmu in env.pmus:
    try:
        subgraph_nodes = list(env.pdcs.union({pmu, env.cc}))
        subgraph = G.subgraph(subgraph_nodes)
        path = nx.shortest_path(subgraph, source=pmu, target=env.cc, weight='latency')
        final_paths.append(path)
    except nx.NetworkXNoPath:
        continue

# Chiamata corretta alla funzione
plot_solution_graph(G, env.pmus, env.cc, env.pdcs, final_paths)


# === PLOT ===
plt.plot(episode_rewards)
plt.xlabel("Episodio")
plt.ylabel("Reward Totale")
plt.title("Andamento Reward")
plt.grid()
plt.show()




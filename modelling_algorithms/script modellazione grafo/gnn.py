import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

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
        temperature = 0.5
        probs = F.softmax(logits / temperature, dim=0)
        return probs

def compute_path_latency(G, path):
    latency = 0
    for i in range(len(path) - 1):
        edge = G[path[i]][path[i+1]]
        if edge.get("status") != "up":
            return float("inf")
        latency += edge["latency"]
    for node in path:
        if G.nodes[node]["role"] == "candidate":
            if G.nodes[node].get("status") != "online":
                return float("inf")
            latency += G.nodes[node].get("processing", 0)
    return latency

def is_valid_chain(G, chain, pmu, cc):
    if chain[0] not in G.neighbors(pmu): return False
    if chain[-1] not in G.neighbors(cc): return False
    for i in range(len(chain) - 1):
        if not G.has_edge(chain[i], chain[i+1]):
            return False
        if G[chain[i]][chain[i+1]]["status"] != "up":
            return False
    return True

def find_best_paths(G, pmus, cc, pdcs, max_latency):
    valid_paths = {}
    for pmu in pmus:
        best = None
        best_delay = float("inf")
        for target in pdcs:
            try:
                subgraph = G.subgraph(pdcs.union({pmu, cc}))
                path = nx.shortest_path(subgraph, source=pmu, target=cc, weight="latency")
                if not is_valid_chain(G, path[1:-1], pmu, cc): continue
                delay = compute_path_latency(G, path)
                if delay < best_delay:
                    best = path
                    best_delay = delay
            except:
                continue
        if best and best_delay <= max_latency:
            valid_paths[pmu] = (best, best_delay)
    return valid_paths

def train_with_policy_gradient(G, max_latency, episodes=5000):
    role_encoding = {"PMU": [1, 0, 0], "CC": [0, 1, 0], "candidate": [0, 0, 1]}
    for n, d in G.nodes(data=True):
        role = d.get("role", "candidate")
        d["x"] = role_encoding.get(role, [0, 0, 1])

    data = from_networkx(G)
    data.x = torch.tensor([G.nodes[n]["x"] for n in G.nodes()], dtype=torch.float)
    pmus = [n for n, d in G.nodes(data=True) if d["role"] == "PMU"]
    cc = [n for n, d in G.nodes(data=True) if d["role"] == "CC"][0]
    model = GraphPolicyNetwork(in_channels=3, hidden_channels=16)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_pdcs = set()
    best_latency = float('inf')
    episode_rewards = []

    for episode in range(episodes):
        pdcs = set()
        log_probs = []
        rewards = []
        valid_episode = True

        for _ in range(15):
            probs = model(data.x, data.edge_index)
            mask = torch.ones_like(probs)

            for i, n in enumerate(G.nodes()):
                node = G.nodes[n]
                if node["role"] in {"PMU", "CC"} or n in pdcs:
                    mask[i] = 0
                elif pdcs:
                    if not any(neigh in pdcs or neigh in pmus or neigh == cc for neigh in G.neighbors(n)):
                        mask[i] = 0
                if node.get("status") != "online":
                    mask[i] = 0

            masked_probs = probs * mask
            total = masked_probs.sum()

            if total.item() <= 0.0 or torch.isnan(total):
                valid_episode = False
                break

            masked_probs = masked_probs / total
            m = torch.distributions.Categorical(masked_probs)
            action_idx = m.sample()
            action_node = list(G.nodes())[action_idx.item()]
            pdcs.add(action_node)
            log_probs.append(m.log_prob(action_idx))

        if not valid_episode:
            continue

        best_paths = find_best_paths(G, pmus, cc, pdcs, max_latency)
        if len(best_paths) < len(pmus):
            reward = -100
        else:
            total_delay = sum(delay for _, delay in best_paths.values())
            reward = -total_delay - len(pdcs) * 3

            if total_delay < best_latency:
                best_latency = total_delay
                best_pdcs = pdcs.copy()

        episode_rewards.append(reward)

        if episode < 10: continue

        loss = -torch.stack(log_probs).mul(torch.tensor(reward)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 1000 == 0:
            print(f"[Ep {episode+1}] Reward: {reward:.2f}  Latency: {best_latency:.2f}  PDCs: {len(best_pdcs)}")

    # Stampa finale
    print("\n✅ Migliori PDC selezionati:", best_pdcs)
    final_paths = find_best_paths(G, pmus, cc, best_pdcs, max_latency)
    for pmu, (path, delay) in final_paths.items():
        print(f"  {pmu} → CC: {path} | Delay = {delay:.2f} ms")

    return best_pdcs

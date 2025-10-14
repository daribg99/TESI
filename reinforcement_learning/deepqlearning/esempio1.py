import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# === Impostazioni ===
nodes_pdc = [1, 2, 3, 4]
state_dim = len(nodes_pdc)
num_actions = len(nodes_pdc)
max_latency = 3
device = torch.device("cpu")

# === Grafo ===
G = nx.Graph()
edges = [
    ("PMU1", 1, 1),
    (1, 2, 2),
    (2, 3, 2),
    (3, "CC", 2),
    ("PMU2", 2, 1),
    (3, 4, 2),
    (4, "CC", 3),
    ("PMU3", 4, 2)
]
for u, v, latency in edges:
    G.add_edge(u, v, latency=latency)

# === Funzione di copertura ===
def coverage_score(pdc_nodes):
    score = 0
    for pmu in ["PMU1", "PMU2", "PMU3"]:
        covered = False
        for pdc in pdc_nodes:
            try:
                path1 = nx.shortest_path(G, source=pmu, target=pdc, weight="latency")
                path2 = nx.shortest_path(G, source=pdc, target="CC", weight="latency")

                latency1 = sum(G[path1[i]][path1[i+1]]["latency"] for i in range(len(path1) - 1))
                latency2 = sum(G[path2[i]][path2[i+1]]["latency"] for i in range(len(path2) - 1))
                total_latency = latency1 + latency2

                if total_latency <= max_latency:
                    covered = True
                    break
            except:
                continue
        if covered:
            score += 1
    return score




# === Rete neurale ===
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# === Parametri DQN ===
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.05
decay = 0.998
alpha = 0.01
episodes = 5000
max_steps = 10

policy_net = DQN(state_dim, num_actions).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# === Training ===
for ep in range(episodes):
    state = np.zeros(state_dim, dtype=np.float32)

    for step in range(max_steps):
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            with torch.no_grad():
                q_values = policy_net(torch.tensor(state, device=device))
                action = torch.argmax(q_values).item()

        next_state = state.copy()
        already_active = state[action] == 1
        next_state[action] = 1

        pdc_nodes_active = [nodes_pdc[i] for i, v in enumerate(next_state) if v == 1]
        covered = coverage_score(pdc_nodes_active)

        if covered == 3:
            print(f"Arrivato con 3 PMU coperti in episodio {ep}, step {step}")
            reward = 50 - 10 * len(pdc_nodes_active)
        elif covered == 2:
            reward = 5
        elif covered == 1:
            reward = 1
        else:
            reward = -5

        if already_active:
            reward -= 2

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        next_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
        q_values = policy_net(state_tensor)
        with torch.no_grad():
            max_next_q = torch.max(policy_net(next_tensor)).item()

        target = q_values.clone()
        target[action] = reward + gamma * max_next_q

        loss = loss_fn(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if covered == 3:
            break

    epsilon = max(min_epsilon, epsilon * decay)

# === Politica finale ===
print("\nPolitica appresa (azioni migliori per ogni stato):")
print("Stato bin |  Nodo 1  |  Nodo 2  |  Nodo 3  |  Nodo 4  |  Azione migliore")
print("-------------------------------------------------------------------------")
for i in range(2**state_dim):
    bin_state = format(i, f"0{state_dim}b")
    state_array = np.array([int(x) for x in bin_state], dtype=np.float32)
    with torch.no_grad():
        q_vals = policy_net(torch.tensor(state_array, device=device))
    best_action = torch.argmax(q_vals).item()
    values = " | ".join(f"{v:7.2f}" for v in q_vals.cpu().numpy())
    print(f"  {bin_state}   | {values} |     Nodo {nodes_pdc[best_action]}")

# === Segui la policy da stato 0000 ===
def get_policy_sequence(policy_net, nodes_pdc):
    visited_states = set()
    state = np.zeros(len(nodes_pdc), dtype=np.float32)
    sequence = []

    while True:
        state_tuple = tuple(state)
        if state_tuple in visited_states:
            break
        visited_states.add(state_tuple)

        active_nodes = [nodes_pdc[i] for i, v in enumerate(state) if v == 1]
        sequence.append((state.copy(), active_nodes, coverage_score(active_nodes)))

        if coverage_score(active_nodes) == 3:
            break

        with torch.no_grad():
            q_vals = policy_net(torch.tensor(state, device=device))
        action = torch.argmax(q_vals).item()
        if state[action] == 1:
            break
        state[action] = 1

    return sequence

sequence = get_policy_sequence(policy_net, nodes_pdc)
print("\n>>> Sequenza appresa dal DQN:")
for i, (state, active, coverage) in enumerate(sequence):
    bits = "".join(str(int(x)) for x in state)
    print(f"Step {i}: Stato {bits} -> PDC attivi: {active} -> Coverage: {coverage}")

# === Verifica configurazioni valide ===
print("\n>>> Tutte le configurazioni valide (coprono 3 PMU):")
found = False
for i in range(1, 2**state_dim):
    state = [int(b) for b in format(i, f"0{state_dim}b")]
    active = [nodes_pdc[j] for j, v in enumerate(state) if v == 1]
    cov = coverage_score(active)
    if cov == 3:
        found = True
        print(f"PDC su {active} (stato {''.join(map(str, state))}) -> coverage = {cov}")
if not found:
    print("Nessuna configurazione trovata.")

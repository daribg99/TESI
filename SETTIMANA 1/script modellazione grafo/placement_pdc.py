import networkx as nx
import random
import numpy as np
from itertools import islice

def place_pdcs_greedy(G, max_latency):
    pdcs = set()
    pmu_paths = {}

    pmu_nodes = [n for n in G.nodes if G.nodes[n].get("role") == "PMU"]

    pmu_to_path = {}

    for pmu in pmu_nodes:
        try:
            paths = nx.shortest_simple_paths(G, source=pmu, target="CC", weight="latency")
            for path in paths:
                # Check validità: tutti i nodi online e archi up
                if all(G.nodes[n].get("status") == "online" for n in path) and \
                   all(G[u][v].get("status") == "up" for u, v in zip(path, path[1:])):
                    pmu_to_path[pmu] = path
                    break
            else:
                print(f"Nessun path valido trovato da {pmu} a CC.")
        except nx.NetworkXNoPath:
            print(f"Nessun path esistente da {pmu} a CC.")
            continue

    for path in pmu_to_path.values(): # trova i PDC
        for node in path[1:-1]:  # Escludi PMU e CC
            if G.nodes[node].get("role") not in {"PMU", "CC"}:
                pdcs.add(node)

    # Calcola i ritardi
    for pmu, path in pmu_to_path.items():
        total_delay = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_delay += G[u][v]["latency"]

        for node in path:
            if node in pdcs:
                total_delay += G.nodes[node].get("processing", 0)

        pmu_paths[pmu] = {"path": path, "delay": total_delay}
    if not pmu_paths:
        print("⚠️ Nessun path valido per alcun PMU.")
        return pdcs

    max_pmu = max(pmu_paths.items(), key=lambda x: x[1]["delay"])
    max_delay = max_pmu[1]["delay"]
    max_path = max_pmu[1]["path"]

    print("\n Path PMU → CC e relative latenze:")
    for pmu, data in pmu_paths.items():
        print(f"  {pmu} → CC: {data['path']} | Ritardo totale = {data['delay']:.2f} ms")

    print()
    if max_delay > max_latency:
        print(f" Ritardo massimo {max_delay:.2f} ms supera la soglia {max_latency} ms.")
        print(f" Causato dal path: {max_pmu[0]} → CC = {max_path}")
    else:
        print(f" Ritardo massimo {max_delay:.2f} ms sotto la soglia {max_latency} ms.")

    return pdcs



def place_pdcs_random(G, max_latency, seed=None):
    if seed is not None:
        random.seed(seed)

    pdcs = set()
    pmu_paths = {}
    pmu_nodes = [n for n in G.nodes if G.nodes[n].get("role") == "PMU"]

    def dfs_random_path(current, target, visited):
        if current == target:
            return [current]

        visited.add(current)
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited and G.nodes[neighbor].get("role") != "PMU":
                if G.nodes[neighbor].get("status") != "online":
                    continue
                if G[current][neighbor].get("status") != "up":
                    continue

                path = dfs_random_path(neighbor, target, visited)
                if path:
                    return [current] + path

        visited.remove(current)
        return None

    for pmu in pmu_nodes:
        path = dfs_random_path(pmu, "CC", set())
        if path is None:
            print(f" Nessun cammino valido trovato da {pmu} a CC (senza PMU/offline/down).")
            continue
        total_delay = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_delay += G[u][v]["latency"]
            if G.nodes[u].get("role") == "candidate":
                total_delay += G.nodes[u].get("processing", 0)

        pmu_paths[pmu] = {"path": path, "delay": total_delay}
        for node in path[1:-1]:
            pdcs.add(node)

    # Trova il path con ritardo massimo
    if pmu_paths:
        max_pmu = max(pmu_paths.items(), key=lambda x: x[1]["delay"])
        max_delay = max_pmu[1]["delay"]
        max_path = max_pmu[1]["path"]

        print("\n Cammini PMU → CC e ritardi:")
        for pmu, data in pmu_paths.items():
            print(f"  {pmu} → CC: {data['path']} | Ritardo = {data['delay']:.2f} ms")

        print()
        if max_delay > max_latency:
            print(f" Ritardo massimo {max_delay:.2f} ms supera la soglia {max_latency} ms.")
            print(f"   Causato dal path: {max_pmu[0]} → CC = {max_path}")
        else:
            print(f" Ritardo massimo {max_delay:.2f} ms sotto la soglia {max_latency} ms.")
    else:
        print(" Nessun path valido trovato da alcun PMU al CC.")

    return pdcs

def q_learning_placement(G, max_latency, episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.8, seed=None, verbose=False):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes_pdc = [n for n, data in G.nodes(data=True)
                 if data.get("role") == "candidate" and data.get("status") == "online"]
    pmu_nodes = [n for n, data in G.nodes(data=True) if data.get("role") == "PMU"]

    num_actions = len(nodes_pdc)
    Q = np.zeros((2 ** num_actions, num_actions)) # inizializzazione Q-table

    def compute_total_delay(path): # calcola il ritardo totale di un path
        delay = 0.0
        for u, v in zip(path, path[1:]):
            delay += G[u][v]["latency"]
        for node in path[1:-1]:
            delay += G.nodes[node].get("processing", 0)
        return delay

    def valid_path(path, state): # controlla che ci siano solo PDC online e archi up
        for node in path[1:-1]:
            if node not in nodes_pdc:
                return False
            idx = nodes_pdc.index(node)
            if state[idx] != 1: # se nodo è candidato ma non attivo ( da state[idx] )
                return False
        for u, v in zip(path, path[1:]): # se qualche arco nel path non è attivo
            if G[u][v].get("status") != "up":
                return False
        if any(G.nodes[n].get("status") != "online" for n in path): # qualche nodo non attivo
            return False
        return True

    def find_best_paths(state):
        pmu_to_best = {} # dizionario nella forma {PMU: (path, delay)}
        for pmu in pmu_nodes:
            best_path = None
            best_delay = float("inf")
            try:
                paths = islice(nx.all_simple_paths(G, pmu, "CC", cutoff=10), 150) # trova fino a 100 cammini, cutoff a 10 nodi per evitare cicli infiniti
                for path in paths:
                    if not valid_path(path, state):
                        continue
                    delay = compute_total_delay(path)
                    if delay < best_delay:
                        best_delay = delay
                        best_path = path
            except nx.NetworkXNoPath:
                continue
            if best_path:
                pmu_to_best[pmu] = (best_path, best_delay)
        return pmu_to_best

    def state_to_index(state):
        return int("".join(str(b) for b in state), 2)

    best_state = None
    best_score = -float("inf")

    for ep in range(episodes):
        state = [0] * num_actions
        pmu_covered = set()

        for step in range(num_actions): # ad ogni step, si attiva un PDC
            s_idx = state_to_index(state) # converte lo stato in indice per Q-table
            available = [i for i in range(num_actions) if state[i] == 0] # seleziona i PDC ancora spenti
            if not available:
                break

            if random.random() < epsilon:
                action = random.choice(available)
            else:
                q_values = Q[s_idx]
                q_masked = [q if i in available else -np.inf for i, q in enumerate(q_values)]
                action = int(np.argmax(q_masked))

            state[action] = 1

            best_paths = find_best_paths(state)
            new_covered = set(best_paths.keys()) # nuove PMU coperte
            delta_covered = len(new_covered - pmu_covered)
            pmu_covered = new_covered

            total_delay = sum(delay for _, delay in best_paths.values())
            pdc_count = sum(state)

            reward = (
                +10 * delta_covered         # copertura incrementale
                - 0.5 * total_delay / 100   # penalizza latenze alte
                - 2 * pdc_count             # penalizza nodi inutili
            )

            next_s_idx = state_to_index(state)
            Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[next_s_idx]) - Q[s_idx, action])

            if verbose:
                print(f"[Ep{ep:4d}][St{step}] ΔPMU={delta_covered}, delay={total_delay:.1f}, PDCs={pdc_count}, R={reward:.2f}")

            if len(pmu_covered) == len(pmu_nodes): # tutte le PMU sono coperte
                break

        # finito l'episodio, calcola il punteggio per quell'episodio ( il Q-value valuta l'action effettuato )
        final_paths = find_best_paths(state)
        total_delay = sum(d for _, d in final_paths.values())
        pdc_count = sum(state)
        final_score = 1000 * len(final_paths) - total_delay - 20 * pdc_count

        if final_score > best_score:
            best_score = final_score
            best_state = state.copy()

    # Esito finale
    selected_pdc = {nodes_pdc[i] for i, b in enumerate(best_state) if b == 1}
    print("\n✅ Politica appresa:")
    print(f" Stato binario: {''.join(str(b) for b in best_state)}")
    print(f" Nodi PDC selezionati: {selected_pdc}")

    print("\n📡 Cammini PMU → CC (validi con PDC selezionati):")
    final_paths = find_best_paths(best_state)
    for pmu in pmu_nodes:
        if pmu in final_paths:
            path, delay = final_paths[pmu]
            status = "✅ OK" if delay <= max_latency else f"⚠️ Ritardo {delay:.2f} ms > soglia"
            print(f"  {pmu} → CC: {path} | Ritardo = {delay:.2f} ms {status}")
        else:
            print(f"  {pmu} → CC: ❌ Nessun path valido")

    return selected_pdc










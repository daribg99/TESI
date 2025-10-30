import networkx as nx
import random
import numpy as np
from itertools import islice
from itertools import combinations

def place_pdcs_greedy(G, max_latency, flag_splitting=False):
    pdcs = set()
    pmu_paths = {}

    bandwidth_usage = {}  # (u, v) → traffico totale corrente
    edge_flow = {}        # (u, v) → lista di PMU che usano l'arco

    pmu_nodes = [n for n in G.nodes if G.nodes[n].get("role") == "PMU"]
    pmu_to_path = {}
    pdc_to_pmus = {}

    # --- Utility ---
    def path_valid(path, extra_rate=0):
        # Verify if path can handle extra_rate on all edges
        for u, v in zip(path, path[1:]):
            edge = (u, v) if (u, v) in G.edges else (v, u)
            capacity = G.edges[edge].get("bandwidth", float("inf"))
            usage = bandwidth_usage.get(edge, 0)
            if usage + extra_rate > capacity:
                return False
        return True

    def update_bandwidth(path, base_rate, pmu):
        for u, v in zip(path, path[1:]):
            edge = (u, v) if (u, v) in G.edges else (v, u)
            bandwidth_usage[edge] = bandwidth_usage.get(edge, 0) + base_rate
            edge_flow.setdefault(edge, []).append(pmu)
        for node in path[1:-1]:
            if G.nodes[node].get("role") not in {"PMU", "CC"}:
                pdc_to_pmus.setdefault(node, set()).add(pmu)

    def remove_pmu_path(pmu):
        if pmu not in pmu_to_path:
            return
        path = pmu_to_path[pmu]
        base_rate = G.nodes[pmu].get("data_rate", 0)
        for u, v in zip(path, path[1:]):
            edge = (u, v) if (u, v) in G.edges else (v, u)
            bandwidth_usage[edge] = max(0, bandwidth_usage.get(edge, 0) - base_rate)
            if edge in edge_flow and pmu in edge_flow[edge]:
                edge_flow[edge].remove(pmu)
                if not edge_flow[edge]:
                    del edge_flow[edge]
        for node in path[1:-1]:
            if node in pdc_to_pmus and pmu in pdc_to_pmus[node]:
                pdc_to_pmus[node].remove(pmu)
                if not pdc_to_pmus[node]:
                    del pdc_to_pmus[node]
        del pmu_to_path[pmu]

    def find_valid_path(pmu, required_rate):
        try:
            for path in nx.shortest_simple_paths(G, source=pmu, target="CC", weight="latency"):
                if not all(G.nodes[n].get("status") == "online" for n in path):
                    continue
                if not all(G[u][v].get("status") == "up" for u, v in zip(path, path[1:])):
                    continue
                if path_valid(path, extra_rate=required_rate):
                    return path
            return None
        except nx.NetworkXNoPath:
            return None

    # --- Main Loop ---
    for pmu in pmu_nodes:
        base_rate = G.nodes[pmu].get("data_rate", 0)
        path = find_valid_path(pmu, base_rate)

        # --- Handle convergence on PDC ---
        if not flag_splitting and path:
            for node in path[1:-1]:
                if node in pdc_to_pmus:
                    ref_pmu = next(iter(pdc_to_pmus[node]))
                    ref_path = pmu_to_path[ref_pmu]
                    if node in ref_path:
                        idx = ref_path.index(node)
                        shared_tail = ref_path[idx:]
                        new_path = path[:path.index(node)] + shared_tail

                        # Check bandwidth on shared segment
                        if not path_valid(new_path, extra_rate=base_rate):
                            print(f"⚠️ Bandiwidth not sufficient to share path at {node} for {pmu}.")
                            affected_pmus = set(pdc_to_pmus[node])
                            affected_pmus.add(pmu)

                            # Remove pmu paths
                            for a_pmu in affected_pmus:
                                remove_pmu_path(a_pmu)

                            # searching for new common path from node to CC
                            for path_cand in nx.shortest_simple_paths(G, source=node, target="CC", weight="latency"):
                                if not all(G.nodes[n].get("status") == "online" for n in path_cand):
                                    continue
                                if not all(G[u][v].get("status") == "up" for u, v in zip(path_cand, path_cand[1:])):
                                    continue

                                total_rate = sum(G.nodes[p].get("data_rate", 0) for p in affected_pmus)
                                if not path_valid(path_cand, extra_rate=total_rate):
                                    continue

                                all_valid = True
                                pmu_new_paths = {}

                                # Check sub-paths from each PMU to node
                                for p in affected_pmus:
                                    try:
                                        sub_path = nx.shortest_path(G, source=p, target=node, weight="latency")
                                    except nx.NetworkXNoPath:
                                        print(f"❌ {p} can't reach {node}, skip common path.")
                                        all_valid = False
                                        break

                                    if not path_valid(sub_path, extra_rate=G.nodes[p].get("data_rate", 0)):
                                        print(f"❌ Bandwidth not sufficient {p} → {node}.")
                                        all_valid = False
                                        break

                                    pmu_new_paths[p] = sub_path + path_cand[1:]

                                # If all PMU can use the new common path
                                if all_valid:
                                    for p, full_path in pmu_new_paths.items():
                                        pmu_to_path[p] = full_path
                                        update_bandwidth(full_path, G.nodes[p].get("data_rate", 0), p)
                                        print(f"✅ New path for {p}: {full_path}")
                                    break
                            break  
                        else:
                            
                            path = new_path
                            break

        if not path:
            print(f"⚠️ No valid path found for {pmu}, skip.")
            continue

        # Valid path found, update structures
        pmu_to_path[pmu] = path
        update_bandwidth(path, base_rate, pmu)

    # Results calculation
    for path in pmu_to_path.values():
        for node in path[1:-1]:
            if G.nodes[node].get("role") not in {"PMU", "CC"}:
                pdcs.add(node)

    for pmu, path in pmu_to_path.items():
        total_delay = sum(G[u][v]["latency"] for u, v in zip(path, path[1:]))
        for node in path:
            if node in pdcs:
                total_delay += G.nodes[node].get("processing", 0)
        pmu_paths[pmu] = {"path": path, "delay": total_delay}

    # --- Output ---
    if not pmu_paths:
        print("⚠️ No valid paths found from any PMU to CC.")
        return pdcs

    max_pmu = max(pmu_paths.items(), key=lambda x: x[1]["delay"])
    max_delay = max_pmu[1]["delay"]
    max_path = max_pmu[1]["path"]

    print("\n📡 Path PMU → CC and relative latency:")
    for pmu, data in pmu_paths.items():
        print(f"  {pmu} → CC: {data['path']} | Total delay = {data['delay']:.2f} ms")

    if max_delay > max_latency:
        print(f"\n⚠️ Max delay {max_delay:.2f} ms exceed the threshold {max_latency} ms.")
        print(f"Critic Path: {max_pmu[0]} → CC = {max_path}")
    else:
        print(f"\n✅ Max delay {max_delay:.2f} ms under the threshold {max_latency} ms.")

    return pdcs, pmu_paths, max_latency




def place_pdcs_random(G, max_latency, seed=None):
    if seed is not None:
        random.seed(seed)

    pdcs = set()
    pmu_paths = {}
    bandwidth_usage = {}  # (u, v) → traffico cumulativo

    pmu_nodes = [n for n in G.nodes if G.nodes[n].get("role") == "PMU"]

    def dfs_random_path(current, target, visited, data_rate):
        if current == target:
            return [current]

        visited.add(current)
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            if G.nodes[neighbor].get("role") == "PMU":
                continue
            if G.nodes[neighbor].get("status") != "online":
                continue
            if G[current][neighbor].get("status") != "up":
                continue

            edge = (current, neighbor) if (current, neighbor) in G.edges else (neighbor, current)
            capacity = G.edges[edge].get("bandwidth", float("inf"))
            usage = bandwidth_usage.get(edge, 0)
            if usage + data_rate > capacity:
                print(f"⚠️ Arco {current}–{neighbor} saturato: {usage + data_rate} kbps > {capacity} kbps")
                continue

            print(f"✅ Arco {current}–{neighbor} OK: {usage + data_rate} ≤ {capacity} kbps")
            path = dfs_random_path(neighbor, target, visited, data_rate)
            if path:
                return [current] + path

        visited.remove(current)
        return None

    for pmu in pmu_nodes:
        data_rate = G.nodes[pmu].get("data_rate", 0)

        path = dfs_random_path(pmu, "CC", set(), data_rate)
        if path is None:
            print(f"⚠️ Nessun cammino valido per {pmu} → CC (nodi down o banda saturata).")
            continue

        # Calcola ritardo
        total_delay = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_delay += G[u][v]["latency"]
            if G.nodes[u].get("role") == "candidate":
                total_delay += G.nodes[u].get("processing", 0)

            # Aggiorna uso della banda con data_rate della PMU sorgente
            edge = (u, v) if (u, v) in G.edges else (v, u)
            bandwidth_usage[edge] = bandwidth_usage.get(edge, 0) + data_rate
            print(f"📶 Arco {u}–{v} aggiornato: {bandwidth_usage[edge]} / {G.edges[edge]['bandwidth']} kbps")

        pmu_paths[pmu] = {"path": path, "delay": total_delay}
        for node in path[1:-1]:
            if G.nodes[node].get("role") not in {"PMU", "CC"}:
                pdcs.add(node)

    # Trova ritardo massimo
    if pmu_paths:
        max_pmu = max(pmu_paths.items(), key=lambda x: x[1]["delay"])
        max_delay = max_pmu[1]["delay"]
        max_path = max_pmu[1]["path"]

        print("\n🎲 Cammini PMU → CC e ritardi:")
        for pmu, data in pmu_paths.items():
            print(f"  {pmu} → CC: {data['path']} | Ritardo = {data['delay']:.2f} ms")

        print()
        if max_delay > max_latency:
            print(f"⚠️ Ritardo massimo {max_delay:.2f} ms supera la soglia {max_latency} ms.")
            print(f"   Causato dal path: {max_pmu[0]} → CC = {max_path}")
        else:
            print(f"✅ Ritardo massimo {max_delay:.2f} ms sotto la soglia {max_latency} ms.")
    else:
        print("❌ Nessun path valido trovato da alcun PMU al CC.")

    return (pdcs, pmu_paths, max_latency)



from itertools import combinations

def place_pdcs_bruteforce(G, max_latency):
    
    def is_valid_chain(path, pdc_nodes, G):
        if not path:
            return False
        node_role = [G.nodes[n].get("role") for n in path]
        if node_role[0] != "PMU" or node_role[-1] != "CC":
            return False
        for i in range(1, len(path) - 1):
            if path[i] not in pdc_nodes:
                return False
            if not G.nodes[path[i]].get("online", True):
                return False
        for u, v in zip(path, path[1:]):
            if not G.has_edge(u, v) or G[u][v].get("status", "up") != "up":
                return False
        return True

    def compute_path_latency(path, G, pdc_nodes):
        latency = 0
        for u, v in zip(path, path[1:]):
            latency += G[u][v].get("latency", 0)
        for n in path:
            if n in pdc_nodes:
                latency += G.nodes[n].get("processing", 0)
        return latency

    pmu_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "PMU"]
    candidate_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "candidate"]
    cc_node = [n for n, d in G.nodes(data=True) if d.get("role") == "CC"][0]

    best_config = None
    best_total_latency = float('inf')
    best_paths = {}
    best_bandwidth_usage = {}

    for k in range(1, len(candidate_nodes) + 1):
        for pdc_nodes in combinations(candidate_nodes, k):
            all_covered = True
            total_latency = 0
            current_path = {}
            bandwidth_usage = {}

            for pmu in pmu_nodes:
                data_rate = G.nodes[pmu].get("data_rate", 0)

                allowed_nodes = set(pdc_nodes) | {pmu, cc_node}
                subgraph = nx.Graph()
                for u in allowed_nodes:
                    subgraph.add_node(u, **G.nodes[u])
                for u, v in G.edges():
                    if u in allowed_nodes and v in allowed_nodes:
                        if G[u][v].get("status", "up") == "up":
                            subgraph.add_edge(u, v, **G[u][v])

                try:
                    path = nx.shortest_path(subgraph, source=pmu, target=cc_node, weight="latency")
                except nx.NetworkXNoPath:
                    all_covered = False
                    break

                if not is_valid_chain(path, pdc_nodes, G):
                    all_covered = False
                    break

                # Verifica saturazione banda su tutto il path
                path_valid = True
                for u, v in zip(path, path[1:]):
                    edge = (u, v) if (u, v) in G.edges else (v, u)
                    capacity = G.edges[edge].get("bandwidth", float("inf"))
                    usage = bandwidth_usage.get(edge, 0)
                    if usage + data_rate > capacity:
                        path_valid = False
                        break

                if not path_valid:
                    all_covered = False
                    break

                # Aggiorna la banda (solo dopo validazione)
                for u, v in zip(path, path[1:]):
                    edge = (u, v) if (u, v) in G.edges else (v, u)
                    bandwidth_usage[edge] = bandwidth_usage.get(edge, 0) + data_rate

                latency = compute_path_latency(path, G, pdc_nodes)
                total_latency += latency
                current_path[pmu] = {"path": path, "delay": latency}

            if all_covered and total_latency < best_total_latency:
                best_config = list(pdc_nodes)
                best_total_latency = total_latency
                best_paths = current_path
                best_bandwidth_usage = bandwidth_usage.copy()

    if best_config:
        print("\n📍 Migliori path PMU → CC con configurazione PDC ottima:")
        for pmu, data in best_paths.items():
            path = data["path"]
            delay = data["delay"]
            print(f"{pmu} → CC: {' → '.join(path)}, Ritardo = {delay:.2f} ms")
            print(f"Banda usata per ogni arco:")
            for u, v in zip(path, path[1:]):
                edge = (u, v) if (u, v) in G.edges else (v, u)
                usage = best_bandwidth_usage.get(edge) or best_bandwidth_usage.get((edge[1], edge[0]), 0)
                print(f"  {u}–{v}: {usage} kbps")
    else:
        print("❌ Nessuna configurazione valida trovata.")

    return best_config if best_config else [], best_paths, max_latency



def q_learning_placement(G, max_latency, episodes=25000, alpha=0.1, gamma=0.9, epsilon=0.8, seed=None, verbose=False):
    
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
    
    # def valid_path(path, state):# controlla che ci siano solo PDC online e archi up
    #     for node in path[1:-1]:
    #         if node not in nodes_pdc:
    #             return False
    #         idx = nodes_pdc.index(node)
    #         if state[idx] != 1:  # se nodo è candidato ma non attivo ( da state[idx] )
    #             return False
    #     for u, v in zip(path, path[1:]): # se qualche arco nel path non è attivo
    #         if G[u][v].get("status") != "up":
    #             return False
    #     if any(G.nodes[n].get("status") != "online" for n in path):
    #         return False
    #     return True
    
    def valid_path(path, state, bandwidth_usage, data_rate):
        # Verifica nodi intermedi PDC attivi
        for node in path[1:-1]:
            if node not in nodes_pdc:
                return False
            idx = nodes_pdc.index(node)
            if state[idx] != 1:
                return False
        # Verifica stato archi e nodi + saturazione banda
        for u, v in zip(path, path[1:]):
            if G[u][v].get("status") != "up":
                return False
            edge = (u, v) if (u, v) in G.edges else (v, u)
            capacity = G.edges[edge].get("bandwidth", float("inf"))
            usage = bandwidth_usage.get(edge, 0)
            if usage + data_rate > capacity:
                return False
        if any(G.nodes[n].get("status") != "online" for n in path):
            return False
        return True


    
    def find_best_paths(state):
        pmu_to_best = {}
        bandwidth_usage = {}
        for pmu in pmu_nodes:
            data_rate = G.nodes[pmu].get("data_rate", 0)
            best_path = None
            best_delay = float("inf")
            try:
                paths = islice(nx.all_simple_paths(G, pmu, "CC", cutoff=15), 500)
                for path in paths:
                    if not valid_path(path, state, bandwidth_usage, data_rate):
                        continue
                    delay = compute_total_delay(path)
                    if delay < best_delay:
                        best_delay = delay
                        best_path = path
            except nx.NetworkXNoPath:
                continue
            if best_path:
                pmu_to_best[pmu] = {"path": best_path, "delay": best_delay}
                for u, v in zip(best_path, best_path[1:]):
                    edge = (u, v) if (u, v) in G.edges else (v, u)
                    bandwidth_usage[edge] = bandwidth_usage.get(edge, 0) + data_rate
        return pmu_to_best


    def state_to_index(state):
        return int("".join(str(b) for b in state), 2)

    best_state = None
    best_score = -float("inf")

    for ep in range(episodes):
        if ep % 1000 == 0: print(f"🔄 Episodio {ep}/{episodes}...")
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

            total_delay = sum(data["delay"] for data in best_paths.values())

            pdc_count = sum(state)

            reward = (
                +20 * delta_covered
                - 1.0 * total_delay / 100
                - 3 * pdc_count
            )


            next_s_idx = state_to_index(state)
            Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[next_s_idx]) - Q[s_idx, action])

            if verbose:
                print(f"[Ep{ep:4d}][St{step}] ΔPMU={delta_covered}, delay={total_delay:.1f}, PDCs={pdc_count}, R={reward:.2f}")

            # if len(pmu_covered) == len(pmu_nodes): # tutte le PMU sono coperte
            #     break

        # finito l'episodio, calcola il punteggio per quell'episodio ( il Q-value valuta l'action effettuato )
        final_paths = find_best_paths(state)
        total_delay = sum(data["delay"] for data in final_paths.values())
        pdc_count = sum(state)
        latency_avg = total_delay / len(final_paths) if final_paths else float('inf')
        # Conta i PDC attivi ma non usati nei path
        used_pdcs = set()
        for data in final_paths.values():
            path = data["path"]
            for node in path[1:-1]:
                used_pdcs.add(node)

        useless_pdc_count = sum(
            1 for i, b in enumerate(state)
            if b == 1 and nodes_pdc[i] not in used_pdcs
        )

        final_score = (
            1000 * len(final_paths) 
            - 5 * latency_avg 
            - 20 * pdc_count
            - 50 * useless_pdc_count  # penalità forte per sprechi
        )


        #final_score = 1000 * len(final_paths) - total_delay - 20 * pdc_count

        if final_score > best_score:
            best_score = final_score
            best_state = state.copy()

    # Esito finale
    # selected_pdc = {nodes_pdc[i] for i, b in enumerate(best_state) if b == 1}
    # print("\n Politica appresa:")
    # print(f" Stato binario: {''.join(str(b) for b in best_state)}")
    # print(f" Nodi PDC selezionati: {selected_pdc}")
    # 🔎 Ricava i PDC usati nei cammini finali
    used_pdcs = set()
    for data in final_paths.values():
        path = data["path"]
        for node in path[1:-1]:  # solo nodi intermedi tra PMU e CC
            used_pdcs.add(node)


    # 🧼 Pulisci best_state per mantenere solo i PDC usati
    clean_best_state = [0] * len(nodes_pdc)
    for i, n in enumerate(nodes_pdc):
        if best_state[i] == 1 and n in used_pdcs:
            clean_best_state[i] = 1
    best_state = clean_best_state
    
    selected_pdc = {nodes_pdc[i] for i, b in enumerate(best_state) if b == 1}


    print("\n📡 Cammini PMU → CC (validi con PDC selezionati):")
    final_paths = find_best_paths(best_state)
    for pmu in pmu_nodes:
        if pmu in final_paths:
            path = final_paths[pmu]["path"]
            delay = final_paths[pmu]["delay"]

            status = " OK" if delay <= max_latency else f" Ritardo {delay:.2f} ms > soglia"
            print(f"  {pmu} → CC: {path} | Ritardo = {delay:.2f} ms {status}")
        else:
            print(f"  {pmu} → CC:  Nessun path valido")

    return selected_pdc, final_paths, max_latency if selected_pdc else None






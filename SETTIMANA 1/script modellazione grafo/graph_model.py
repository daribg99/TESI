import networkx as nx
from visualizer import draw_graph

def create_graph():
    G = nx.Graph()

    # Nodi
    G.add_node("CC", group="core", level=0, processing=10, memory=32, storage=1000, status="online", energy=1.0)
    G.add_node("P1", group="A", level=1, processing=6, memory=16, storage=500, status="online", energy=0.95)
    G.add_node("P2", group="A", level=1, processing=6, memory=16, storage=500, status="online", energy=0.93)
    G.add_node("R1", group="A", level=2, processing=4, memory=8, storage=250, status="online", energy=0.88)
    G.add_node("R2", group="A", level=2, processing=4, memory=8, storage=250, status="online", energy=0.86)
    G.add_node("L1", group="A", level=3, processing=1, memory=2, storage=64, status="online", energy=0.75)
    G.add_node("L2", group="A", level=3, processing=1, memory=2, storage=64, status="online", energy=0.70)
    G.add_node("L3", group="A", level=3, processing=1, memory=2, storage=64, status="online", energy=0.72)

    # Archi
    G.add_edge("CC", "P1", latency=2, bandwidth=1000, status="up", type="fiber")
    G.add_edge("CC", "P2", latency=3, bandwidth=500, status="up", type="ethernet")
    G.add_edge("P1", "R1", latency=4, bandwidth=200, status="up", type="ethernet")
    G.add_edge("P2", "R2", latency=4, bandwidth=200, status="up", type="ethernet")
    G.add_edge("R1", "L1", latency=6, bandwidth=100, status="up", type="wireless")
    G.add_edge("R1", "L2", latency=7, bandwidth=100, status="up", type="wireless")
    G.add_edge("R2", "L3", latency=5, bandwidth=100, status="up", type="wireless")

    
    
    # Modifica interattiva delle latenze
    while True:
        print("\n🔗 Lista delle latenze attuali:")
        for u, v, data in G.edges(data=True):
            print(f"  {u} – {v}: latenza = {data['latency']} ms")

        risposta = input("\nVuoi modificare una latenza? (s/n): ").lower()
        if risposta != "s":
            break

        u = input("Nodo 1 dell’arco: ").strip()
        v = input("Nodo 2 dell’arco: ").strip()

        if G.has_edge(u, v):
            try:
                nuova_latenza = float(input(f"Inserisci nuova latenza per l’arco {u}–{v}: "))
                G[u][v]["latency"] = nuova_latenza
                print(f"✔️ Latenza aggiornata per {u}–{v} a {nuova_latenza} ms.")
            except ValueError:
                print("❌ Valore non valido. Riprova.")
        else:
            print("❌ L’arco specificato non esiste.")

    return G

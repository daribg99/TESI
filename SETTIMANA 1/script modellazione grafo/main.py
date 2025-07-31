from graph_model import create_graph
from visualizer import draw_graph
from placement_pdc import place_pdcs_greedy
from placement_pdc import place_pdcs_random
from placement_pdc import place_pdcs_bruteforce
from placement_pdc import q_learning_placement
#from placement_pdc import place_pdcs_centrality
#from placement_pdc import place_pdcs_betweenness
from graph_model import modify_latency
from graph_model import modify_edge_status
from gnn import train_with_policy_gradient
# Algoritmi di posizionamento: 
# place_pdcs_greedy(G, max_latency)
# place_pdcs_random(G, num_pdcs, seed=None)
# place_pdcs_centrality(G, num_pdcs)

def choose_algorithm(G):
    print("Scegli un algoritmo di posizionamento PDC:")
    print("1. Greedy (con latenza massima)")
    print("2. Random (con numero di PDC specificato)")
    print("3. Q-Learning")
    print("4. GNN + Policy Gradient")
    print("5. Bruteforce")
    print("6. Esci")

    choice = input("Inserisci il numero dell'algoritmo: ")

    if choice == "1":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        return place_pdcs_greedy(G, max_latency)
    elif choice == "2":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        seed = int(input("Inserisci il seed (lascia vuoto per nessun seed): ") or 42)
        return place_pdcs_random(G, max_latency, seed)
    elif choice == "3":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        return q_learning_placement(G, max_latency)
    elif choice == "4":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        return train_with_policy_gradient(G, max_latency)
    elif choice == "5":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        return place_pdcs_bruteforce(G, max_latency)
    elif choice == "6":
        print("Uscita in corso...")
        exit(0)
    else:
        print("Scelta non valida. Riprova.")
        return choose_algorithm(G)

def main():
    while True:
        G = create_graph(seed=42)
        modify_latency(G)
        modify_edge_status(G)
        (pdcs,path, max_latency) = choose_algorithm(G)
        print("PDC assegnati nei cluster:", pdcs)
        draw_graph(G, pdcs=pdcs,paths=path, max_latency=max_latency)

if __name__ == "__main__":
    main()

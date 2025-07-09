from graph_model import create_graph
from visualizer import draw_graph
from placement_pdc import place_pdcs_greedy
from placement_pdc import place_pdcs_random
from placement_pdc import place_pdcs_centrality
from placement_pdc import place_pdcs_betweenness

# Algoritmi di posizionamento: 
# place_pdcs_greedy(G, max_latency)
# place_pdcs_random(G, num_pdcs, seed=None)
# place_pdcs_centrality(G, num_pdcs)

def choose_algorithm(G):
    print("Scegli un algoritmo di posizionamento PDC:")
    print("1. Greedy (con latenza massima)")
    print("2. Random (con numero di PDC specificato)")
    print("3. Centrality (basato su closeness centrality)")
    print("4. Betweenness (basato su betweenness centrality)")
    print("5. Esci")

    choice = input("Inserisci il numero dell'algoritmo: ")

    if choice == "1":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        return place_pdcs_greedy(G, max_latency)
    elif choice == "2":
        num_pdcs = int(input("Inserisci il numero di PDC da posizionare: "))
        seed = int(input("Inserisci il seed (lascia vuoto per nessun seed): ") or 42)
        return place_pdcs_random(G, num_pdcs, seed)
    elif choice == "3":
        num_pdcs = int(input("Inserisci il numero di PDC da posizionare: "))
        return place_pdcs_centrality(G, num_pdcs)
    elif choice == "4":
        num_pdcs = int(input("Inserisci il numero di PDC da posizionare: "))
        return place_pdcs_betweenness(G, num_pdcs)
    elif choice == "5":
        print("Uscita in corso...")
        exit(0)
    else:
        print("Scelta non valida. Riprova.")
        return choose_algorithm(G)

def main():
    while True:
        G = create_graph()
        pdcs = choose_algorithm(G)
        print("PDC assegnati nei cluster:", pdcs)
        draw_graph(G, pdcs=pdcs)

if __name__ == "__main__":
    main()

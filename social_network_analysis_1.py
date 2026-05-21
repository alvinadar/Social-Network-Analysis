"""
============================================================================
SOCIAL NETWORK ANALYSIS — A Hands-On Tour
============================================================================
A single runnable script that teaches the core concepts of Social Network
Analysis (SNA) through a synthetic but realistic friendship network.

You know Python but are new to networks, so every section explains the
INTUITION first, then shows the code, then prints/plots the result.

Run it:   python social_network_analysis.py
Outputs:  several PNG images saved next to this script.

Concepts covered, in order:
  1. Building a graph & reading its basic structure
  2. Degree, density, paths — the vocabulary of networks
  3. Centrality — four different answers to "who is important?"
  4. Community detection — finding hidden friend groups
  5. Information diffusion — simulating how a rumor/post spreads
============================================================================
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import Counter

# Make results reproducible: same "random" network every run.
random.seed(42)

# A consistent layout seed so node positions don't jump between plots.
LAYOUT_SEED = 7


# ============================================================================
# SECTION 1 — BUILDING A GRAPH
# ============================================================================
# A GRAPH is just NODES (people) connected by EDGES (relationships).
# In an UNDIRECTED graph, friendship is mutual: if A is B's friend, B is A's.
# (Twitter "follows" would be DIRECTED instead — we'll note that later.)
#
# Instead of typing edges by hand, we generate a network with a realistic
# shape using the "caveman + rewire" idea: start with tight-knit clusters
# (like real friend groups / communities), then randomly rewire a few edges
# so the clusters aren't perfectly isolated. This mimics how real social
# networks have dense pockets loosely linked to each other.

def build_social_network():
    # connected_caveman_graph(num_cliques, clique_size):
    #   creates `num_cliques` tightly-connected groups of `clique_size` people.
    G = nx.connected_caveman_graph(5, 8)  # 5 groups of 8 people = 40 people

    # Rewire ~12% of edges randomly. This adds "weak ties" between groups —
    # the bridges that make a social network feel like one connected world
    # rather than 5 separate islands.
    nodes = list(G.nodes())
    edges = list(G.edges())
    for u, v in edges:
        if random.random() < 0.12:
            G.remove_edge(u, v)
            new_target = random.choice(nodes)
            # avoid self-loops and duplicate edges
            while new_target == u or G.has_edge(u, new_target):
                new_target = random.choice(nodes)
            G.add_edge(u, new_target)

    # Give people names instead of bare numbers — easier to read in output.
    names = [f"P{i:02d}" for i in G.nodes()]
    mapping = {i: name for i, name in zip(G.nodes(), names)}
    G = nx.relabel_nodes(G, mapping)
    return G


# ============================================================================
# SECTION 2 — BASIC STRUCTURE: the vocabulary of networks
# ============================================================================
def describe_structure(G):
    print("=" * 70)
    print("SECTION 2 — BASIC NETWORK STRUCTURE")
    print("=" * 70)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"People (nodes):        {n}")
    print(f"Friendships (edges):   {m}")

    # DEGREE = how many friends a person has. The most basic popularity measure.
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / n
    print(f"Average friends/person: {avg_degree:.2f}")

    # DENSITY = actual edges / all possible edges. 1.0 = everyone knows everyone.
    # Real social networks are SPARSE (low density) — you know a tiny fraction
    # of all people.
    print(f"Density:               {nx.density(G):.3f}  (1.0 = everyone connected to everyone)")

    # Is the whole network reachable as one piece?
    if nx.is_connected(G):
        # AVERAGE PATH LENGTH = typical "degrees of separation" between two people.
        # The famous "six degrees" idea — usually surprisingly small.
        print(f"Avg path length:       {nx.average_shortest_path_length(G):.2f}  (typical degrees of separation)")
        print(f"Diameter:              {nx.diameter(G)}  (the two most-distant people)")

    # CLUSTERING COEFFICIENT = "are my friends also friends with each other?"
    # High clustering is a hallmark of social networks (friend groups).
    print(f"Clustering coefficient: {nx.average_clustering(G):.3f}  (do your friends know each other?)")

    # Who has the most friends?
    top = sorted(degrees.items(), key=lambda x: -x[1])[:3]
    print(f"Most-connected people:  {top}")
    print()
    return degrees


# ============================================================================
# SECTION 3 — CENTRALITY: four answers to "who matters?"
# ============================================================================
# "Important" means different things. Each centrality measure captures a
# different kind of importance. This is the heart of influence analysis.
def analyze_centrality(G):
    print("=" * 70)
    print("SECTION 3 — CENTRALITY (who is influential, and why?)")
    print("=" * 70)

    # DEGREE centrality: popularity. Many direct connections.
    degree_c = nx.degree_centrality(G)

    # BETWEENNESS centrality: brokerage. How often you sit ON the shortest
    # path between two other people. These are the BRIDGES — remove them and
    # the network can fragment. Often the real power players: they control
    # the flow of information between groups.
    between_c = nx.betweenness_centrality(G)

    # CLOSENESS centrality: reach. How close you are (on average) to everyone
    # else. High closeness = you can spread something to the whole network fast.
    close_c = nx.closeness_centrality(G)

    # EIGENVECTOR centrality: influence by association. You're important if
    # you're connected to OTHER important people. (This is the seed of the
    # idea behind Google's PageRank.)
    eigen_c = nx.eigenvector_centrality(G, max_iter=1000)

    def top3(d):
        return sorted(d.items(), key=lambda x: -x[1])[:3]

    print("Top 3 by DEGREE (most friends / popularity):")
    for name, val in top3(degree_c):
        print(f"   {name}: {val:.3f}")
    print("\nTop 3 by BETWEENNESS (bridges between groups — info brokers):")
    for name, val in top3(between_c):
        print(f"   {name}: {val:.3f}")
    print("\nTop 3 by CLOSENESS (can reach everyone fastest):")
    for name, val in top3(close_c):
        print(f"   {name}: {val:.3f}")
    print("\nTop 3 by EIGENVECTOR (connected to other influential people):")
    for name, val in top3(eigen_c):
        print(f"   {name}: {val:.3f}")
    print("\nNotice: the 'most popular' person is often NOT the best 'bridge'.")
    print("That gap is one of the most useful insights in network analysis.\n")

    return degree_c, between_c, close_c, eigen_c


def plot_centrality(G, pos, between_c):
    """Visualize betweenness: node size = how much of a 'bridge' each person is."""
    plt.figure(figsize=(11, 8))
    sizes = [3000 * between_c[n] + 50 for n in G.nodes()]
    colors = [between_c[n] for n in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                                   cmap=plt.cm.plasma)
    nx.draw_networkx_edges(G, pos, alpha=0.25)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.colorbar(nodes, label="Betweenness centrality (bridge importance)")
    plt.title("Who are the bridges? Bigger & brighter = controls more information flow")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("02_centrality_betweenness.png", dpi=130)
    plt.close()
    print("   -> saved 02_centrality_betweenness.png\n")


# ============================================================================
# SECTION 4 — COMMUNITY DETECTION: finding hidden groups
# ============================================================================
# A COMMUNITY is a set of people more densely connected to each other than
# to the rest of the network — i.e. a friend group, a department, a fandom.
# We use the Louvain method, which optimizes "modularity": a score for how
# good a particular grouping is (high = dense inside groups, sparse between).
def detect_communities(G):
    print("=" * 70)
    print("SECTION 4 — COMMUNITY DETECTION (finding hidden friend groups)")
    print("=" * 70)

    import community as community_louvain  # the python-louvain package

    partition = community_louvain.best_partition(G, random_state=42)
    modularity = community_louvain.modularity(partition, G)

    communities = Counter(partition.values())
    print(f"Found {len(communities)} communities.")
    print(f"Modularity score: {modularity:.3f}  (>0.3 means clear group structure)")
    print("Community sizes:")
    for comm_id, size in sorted(communities.items()):
        members = [n for n, c in partition.items() if c == comm_id]
        print(f"   Community {comm_id}: {size} people -> {members}")
    print()
    return partition


def plot_communities(G, pos, partition):
    """Color each node by the community it was assigned to."""
    plt.figure(figsize=(11, 8))
    cmap = plt.cm.tab10
    colors = [partition[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=cmap,
                           node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.25)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title("Communities detected by the Louvain method (each color = one group)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("03_communities.png", dpi=130)
    plt.close()
    print("   -> saved 03_communities.png\n")


# ============================================================================
# SECTION 5 — INFORMATION DIFFUSION: how does a post/rumor spread?
# ============================================================================
# The INDEPENDENT CASCADE model: a few "seed" people start out informed.
# Each informed person gets ONE chance to infect each uninformed neighbor,
# succeeding with probability p. We track how the "infection" grows in waves.
#
# This is how you'd model a viral post, a rumor, or word-of-mouth adoption.
# Choosing good seeds (e.g. high-centrality people) makes the spread bigger —
# that's the core idea behind "influence maximization" in viral marketing.
def simulate_diffusion(G, seeds, p=0.15, max_steps=15):
    print("=" * 70)
    print("SECTION 5 — INFORMATION DIFFUSION (simulating a viral spread)")
    print("=" * 70)
    print(f"Seeds (who starts the rumor): {seeds}")
    print(f"Transmission probability per friend: {p}")

    informed = set(seeds)
    newly_informed = set(seeds)
    history = [len(informed)]

    step = 0
    while newly_informed and step < max_steps:
        step += 1
        next_wave = set()
        for person in newly_informed:
            for neighbor in G.neighbors(person):
                if neighbor not in informed:
                    if random.random() < p:  # the coin flip per edge
                        next_wave.add(neighbor)
        informed |= next_wave
        newly_informed = next_wave
        history.append(len(informed))
        print(f"   Step {step:2d}: {len(next_wave):2d} new people reached "
              f"-> {len(informed)} total ({100*len(informed)/G.number_of_nodes():.0f}% of network)")

    print(f"\nFinal reach: {len(informed)}/{G.number_of_nodes()} people "
          f"({100*len(informed)/G.number_of_nodes():.0f}%)\n")
    return history


def compare_seed_strategies(G, between_c):
    """Show why WHO you seed matters: best bridges vs random people."""
    print("Comparing seeding strategies (avg reach over 200 simulations each):")

    # Strategy A: seed the 2 best 'bridge' people (highest betweenness).
    best_bridges = [n for n, _ in sorted(between_c.items(), key=lambda x: -x[1])[:2]]

    def avg_reach(seeds, trials=200):
        total = 0
        for _ in range(trials):
            informed = set(seeds)
            wave = set(seeds)
            while wave:
                nxt = set()
                for person in wave:
                    for nb in G.neighbors(person):
                        if nb not in informed and random.random() < 0.15:
                            nxt.add(nb)
                informed |= nxt
                wave = nxt
            total += len(informed)
        return total / trials

    smart = avg_reach(best_bridges)

    random_reaches = []
    all_nodes = list(G.nodes())
    for _ in range(20):  # try 20 different random pairs, average them
        random_reaches.append(avg_reach(random.sample(all_nodes, 2), trials=20))
    rnd = sum(random_reaches) / len(random_reaches)

    print(f"   Seeding 2 BEST bridges {best_bridges}: reaches ~{smart:.1f} people")
    print(f"   Seeding 2 RANDOM people (averaged):       reaches ~{rnd:.1f} people")
    print(f"   Smart seeding is ~{100*(smart-rnd)/rnd:.0f}% more effective.\n")


def plot_diffusion(history):
    """Plot the classic S-curve of adoption over time."""
    plt.figure(figsize=(9, 5.5))
    plt.plot(range(len(history)), history, marker="o", linewidth=2, color="#d6336c")
    plt.fill_between(range(len(history)), history, alpha=0.15, color="#d6336c")
    plt.xlabel("Time step (wave of spreading)")
    plt.ylabel("Total people reached")
    plt.title("Information diffusion: the classic 'S-curve' of viral spread")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_diffusion_curve.png", dpi=130)
    plt.close()
    print("   -> saved 04_diffusion_curve.png\n")


def plot_overview(G, pos, degrees):
    """A first look at the whole network: node size = number of friends."""
    plt.figure(figsize=(11, 8))
    sizes = [80 * degrees[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4c6ef5", alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.25)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title("The social network — node size = number of friends (degree)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("01_network_overview.png", dpi=130)
    plt.close()
    print("   -> saved 01_network_overview.png\n")


# ============================================================================
# MAIN — run the whole tour
# ============================================================================
def main():
    G = build_social_network()

    # One shared layout so every plot shows nodes in the same place.
    pos = nx.spring_layout(G, seed=LAYOUT_SEED, k=0.4)

    degrees = describe_structure(G)
    plot_overview(G, pos, degrees)

    degree_c, between_c, close_c, eigen_c = analyze_centrality(G)
    plot_centrality(G, pos, between_c)

    partition = detect_communities(G)
    plot_communities(G, pos, partition)

    # Seed the diffusion with the single best 'bridge' person.
    best_bridge = max(between_c, key=between_c.get)
    history = simulate_diffusion(G, seeds=[best_bridge])
    plot_diffusion(history)
    compare_seed_strategies(G, between_c)

    print("=" * 70)
    print("DONE. Four images were saved. Open them to SEE each concept:")
    print("   01_network_overview.png    — the whole network")
    print("   02_centrality_betweenness.png — the bridges/brokers")
    print("   03_communities.png         — hidden friend groups")
    print("   04_diffusion_curve.png     — viral spread over time")
    print("=" * 70)


if __name__ == "__main__":
    main()

import random
import argparse

def generate_graph(num_vertices, num_edges, output_file):
    """
    Generate a graph with the specified number of vertices and edges.
    Ensures no duplicate edges or self-loops.
    """
    if num_edges > num_vertices * (num_vertices - 1) // 2:
        raise ValueError("Too many edges for the given number of vertices.")

    pairs = set()
    edges = []
    while len(pairs) < num_edges:
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        if u != v:
            pair = (min(u, v), max(u, v))  # undirected key
            if pair not in pairs:
                weight = round(random.uniform(1.0, 100.0), 2)
                pairs.add(pair)
                edges.append((u, v, weight))

    # Write edges to the output file
    with open(output_file, "w") as f:
        for u, v, weight in edges:
            f.write(f"{u} {v} {weight}\n")

    print(f"Graph with {num_vertices} vertices and {num_edges} edges written to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Dijkstra-ready graph and save to a .edges file.")
    parser.add_argument("num_vertices", type=int, help="Number of vertices in the graph.")
    parser.add_argument("num_edges", type=int, help="Number of edges in the graph.")
    parser.add_argument("output_file", type=str, help="Output file to save the graph.")
    args = parser.parse_args()

    generate_graph(args.num_vertices, args.num_edges, args.output_file)
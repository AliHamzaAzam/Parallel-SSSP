#!/usr/bin/env python3
"""
visualize_graph.py

Read a weighted directed-edges file (u v w per line), build a Graphviz digraph,
and write a .dot file and optionally render to PNG.
Usage:
  python3 visualize_graph.py input.edges -o output_prefix
This will create output_prefix.dot (and output_prefix.png if --render).
"""
import argparse
import sys

try:
    from graphviz import Graph  # undirected graph  # type: ignore
except ImportError:
    sys.exit('Required package graphviz not installed. Run: pip install graphviz')


def parse_edges(path):
    edges = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            u, v, w = parts[:3]
            edges.append((u, v, w))
    return edges

def parse_updates(path):
    updates = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'): continue
            parts = ln.split()
            if parts[0] == 'd' and len(parts) >= 3:
                updates.append(('d', parts[1], parts[2], None))
            elif parts[0] == 'i' and len(parts) >= 4:
                updates.append(('i', parts[1], parts[2], parts[3]))
    return updates

def compute_sssp(edges, source=0):
    """Compute shortest-path tree parents for directed weighted graph."""
    import heapq
    # Build adjacency list
    adj = {}
    nodes = set()
    for u, v, w in edges:
        ui, vi = int(u), int(v)
        wi = float(w)
        # undirected: add both directions
        adj.setdefault(ui, []).append((vi, wi))
        adj.setdefault(vi, []).append((ui, wi))
        nodes.add(ui); nodes.add(vi)
    # Initialize distances and parents
    dist = {u: float('inf') for u in nodes}
    parent = {u: None for u in nodes}
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return parent, dist

def main():
    p = argparse.ArgumentParser(description='Visualize a weighted directed graph from an .edges file')
    p.add_argument('input', help='Input .edges file (u v w per line)')
    p.add_argument('-o', '--output', required=True, help='Output file prefix')
    p.add_argument('--updates', help='Optional updates file (d/u v or i u v w per line)')
    p.add_argument('--render', action='store_true', help='Also render PNG using Graphviz')
    args = p.parse_args()

    edges = parse_edges(args.input)
    # Compute shortest-path tree for original graph
    parent0, dist0 = compute_sssp(edges, source=0)

    # if updates provided, build two graphs and exit
    if args.updates:
        # apply updates to a copy
        edges_updated = list(edges)
        for typ,u,v,w in parse_updates(args.updates):
            if typ == 'd':
                # Remove undirected edge in both orientations
                edges_updated = [e for e in edges_updated if not ((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u))]
            elif typ == 'i':
                edges_updated.append((u, v, w))
        # Compute shortest-path tree for updated graph
        parent1, dist1 = compute_sssp(edges_updated, source=0)
        # write original
        g0 = Graph(comment=f'Original from {args.input}', format='png')
        g0.attr(rankdir='LR'); g0.node_attr.update(shape='circle')
        # annotate each vertex with its computed min-cost
        for n, c in dist0.items():
            g0.node(str(n), label=f"{n}\n{c}")
        for u, v, w in edges:
            u_i, v_i = int(u), int(v)
            # highlight undirected tree edges in either direction
            if parent0.get(v_i) == u_i or parent0.get(u_i) == v_i:
                g0.edge(u, v, label=w, color='blue', penwidth='3')
            else:
                g0.edge(u, v, label=w, color='gray')
        dot0 = args.output + '_original.dot'; g0.save(dot0); print(f'Written original DOT: {dot0}')
        if args.render: out0 = g0.render(filename=args.output + '_originagit status --porcelainl', cleanup=False); print(f'Rendered original PNG: {out0}')
        # write updated
        g1 = Graph(comment=f'Updated from {args.updates}', format='png')
        g1.attr(rankdir='LR'); g1.node_attr.update(shape='circle')
        for n, c in dist1.items():
            g1.node(str(n), label=f"{n}\n{c}")
        for u, v, w in edges_updated:
            u_i, v_i = int(u), int(v)
            # highlight undirected updated tree edges
            if parent1.get(v_i) == u_i or parent1.get(u_i) == v_i:
                g1.edge(u, v, label=w, color='red', penwidth='3')
            else:
                g1.edge(u, v, label=w, color='gray')
        dot1 = args.output + '_updated.dot'; g1.save(dot1); print(f'Written updated DOT: {dot1}')
        if args.render: out1 = g1.render(filename=args.output + '_updated', cleanup=False); print(f'Rendered updated PNG: {out1}')
        return

    g = Graph(comment=f'Graph from {args.input}', format='png')
    g.attr(rankdir='LR'); g.node_attr.update(shape='circle')
    for n, c in dist0.items():
        g.node(str(n), label=f"{n}\n{c}")
    for u, v, w in edges:
        u_i, v_i = int(u), int(v)
        # highlight undirected tree edges in either direction
        if parent0.get(v_i) == u_i or parent0.get(u_i) == v_i:
            g.edge(u, v, label=str(w), color='blue', penwidth='3')
        else:
            g.edge(u, v, label=str(w), color='gray', penwidth='1')

    dot_path = args.output + '.dot'
    g.save(dot_path)
    print(f'Written DOT file: {dot_path}')

    if args.render:
        out = g.render(filename=args.output, cleanup=False)
        print(f'Rendered PNG: {out}')

if __name__ == '__main__':
    main()

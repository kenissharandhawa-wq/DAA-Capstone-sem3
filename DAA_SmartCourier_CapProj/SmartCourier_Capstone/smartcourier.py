import csv, heapq, time, random, argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent

def load_parcels(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    parcels = []
    for r in rows:
        p = {'id': r['id'], 'destination': r['destination'], 'weight': int(r['weight']), 'priority': int(r['priority']), 'deadline': int(r['deadline'])}
        if p['weight'] <= 0 or p['priority'] <= 0 or p['deadline'] <= 0:
            raise ValueError('Weights, priorities and deadlines must be positive')
        parcels.append(p)
    if len({p['id'] for p in parcels}) != len(parcels):
        raise ValueError('Parcel IDs must be unique')
    return parcels

def load_graph(path):
    graph = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            a, b, w = row['from'], row['to'], float(row['distance'])
            if w < 0: raise ValueError('Dijkstra requires nonnegative distances')
            graph.setdefault(a, {})[b] = min(w, graph.get(a, {}).get(b, float('inf')))
            graph.setdefault(b, {})[a] = min(w, graph.get(b, {}).get(a, float('inf')))
    return graph

def merge_sort(items, key):
    if len(items) <= 1: return items[:]
    mid = len(items)//2
    a, b = merge_sort(items[:mid], key), merge_sort(items[mid:], key)
    out = []; i = j = 0
    while i < len(a) and j < len(b):
        if key(a[i]) <= key(b[j]): out.append(a[i]); i += 1
        else: out.append(b[j]); j += 1
    return out + a[i:] + b[j:]

def parcel_value(p):
    return p['priority'] * 100 + max(0, 50 - p['deadline'])

def knapsack(parcels, capacity):
    if capacity < 0: raise ValueError('Capacity must be nonnegative')
    n = len(parcels)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i, p in enumerate(parcels, 1):
        for w in range(capacity+1):
            dp[i][w] = dp[i-1][w]
            if p['weight'] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-p['weight']] + parcel_value(p))
    selected = []; w = capacity
    for i in range(n, 0, -1):
        p = parcels[i-1]
        if p['weight'] <= w and dp[i][w] == dp[i-1][w-p['weight']] + parcel_value(p) and dp[i][w] > dp[i-1][w]:
            selected.append(p); w -= p['weight']
    selected.reverse()
    return selected, dp[n][capacity], dp

def dijkstra(graph, start):
    if start not in graph: raise ValueError(f'Unknown node: {start}')
    dist = {v: float('inf') for v in graph}; prev = {}
    dist[start] = 0; heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]: continue
        for v, w in graph[u].items():
            nd = d + w
            if nd < dist[v]: dist[v] = nd; prev[v] = u; heapq.heappush(heap, (nd, v))
    return dist, prev

def shortest_path(prev, start, target):
    if start == target: return [start]
    if target not in prev: return []
    path = [target]
    while path[-1] != start:
        if path[-1] not in prev: return []
        path.append(prev[path[-1]])
    return list(reversed(path))

def nearest_neighbor_route(graph, depot, destinations):
    targets = set(destinations) - {depot}
    if depot not in graph or any(x not in graph for x in targets):
        raise ValueError('Unknown delivery location in graph')
    relevant = [depot] + sorted(targets)
    shortest = {u: dijkstra(graph, u) for u in relevant}
    route = [depot]; total = 0.0; current = depot; legs = []
    while targets:
        nxt = min(targets, key=lambda x: (shortest[current][0][x], x))
        d = shortest[current][0][nxt]
        if d == float('inf'): raise ValueError(f'No path from {current} to {nxt}')
        legs.append((current, nxt, d, shortest_path(shortest[current][1], current, nxt)))
        total += d; route.append(nxt); targets.remove(nxt); current = nxt
    back = shortest[current][0][depot]
    if back == float('inf'): raise ValueError('Cannot return to depot')
    legs.append((current, depot, back, shortest_path(shortest[current][1], current, depot)))
    total += back; route.append(depot)
    return route, total, legs

def plan(parcels, graph, capacity, depot='Depot'):
    start = time.perf_counter(); sorted_parcels = merge_sort(parcels, lambda p: (-p['priority'], p['deadline'], p['id'])); sort_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter(); selected, score, dp = knapsack(sorted_parcels, capacity); pack_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter(); route, distance, legs = nearest_neighbor_route(graph, depot, [p['destination'] for p in selected]); route_ms = (time.perf_counter()-start)*1000
    return {'sorted': sorted_parcels, 'selected': selected, 'score': score, 'dp': dp, 'route': route, 'distance': distance, 'legs': legs, 'times': {'Sorting': sort_ms, 'Knapsack DP': pack_ms, 'Routing': route_ms}}

def benchmarks(graph, outdir):
    rng = random.Random(42); sizes = [10, 30, 60, 100, 160]
    nodes = [n for n in graph if n != 'Depot']; records = []
    for n in sizes:
        ps = [{'id': f'P{i}', 'destination': rng.choice(nodes), 'weight': rng.randint(1, 10), 'priority': rng.randint(1, 5), 'deadline': rng.randint(1, 15)} for i in range(n)]
        timings = {}
        for label, fn in [('Sorting', lambda: merge_sort(ps, lambda p: (-p['priority'], p['deadline'], p['id']))), ('Knapsack DP', lambda: knapsack(ps, 100)), ('Routing', lambda: nearest_neighbor_route(graph, 'Depot', [p['destination'] for p in ps]))]:
            t0 = time.perf_counter()
            for _ in range(5): fn()
            timings[label] = (time.perf_counter()-t0)*1000/5
        records.append({'Parcels': n, **timings})
    with open(outdir/'runtime_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Parcels', 'Sorting', 'Knapsack DP', 'Routing']); writer.writeheader(); writer.writerows(records)
    for label in ['Sorting', 'Knapsack DP', 'Routing']:
        plt.plot(sizes, [r[label] for r in records], marker='o', label=label)
    plt.xlabel('Number of parcels'); plt.ylabel('Average runtime (ms)'); plt.title('SmartCourier Algorithm Runtime Comparison'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(outdir/'runtime_chart.png', dpi=170); plt.close()
    return records

def main():
    ap = argparse.ArgumentParser(description='SmartCourier: intelligent parcel planning')
    ap.add_argument('--parcels', default=str(BASE/'data'/'parcels.csv'))
    ap.add_argument('--graph', default=str(BASE/'data'/'roads.csv'))
    ap.add_argument('--capacity', type=int, default=15)
    ap.add_argument('--depot', default='Depot')
    ap.add_argument('--output', default=str(BASE/'output'))
    args = ap.parse_args(); outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    parcels = load_parcels(args.parcels); graph = load_graph(args.graph)
    result = plan(parcels, graph, args.capacity, args.depot)
    lines = ['SMARTCOURIER - DELIVERY PLANNING RESULT', '='*46, f'Input parcels: {len(parcels)}', f'Vehicle capacity: {args.capacity} kg', '', 'PRIORITIZED PARCELS']
    lines += [f"{p['id']}: {p['destination']}, {p['weight']} kg, priority {p['priority']}, deadline {p['deadline']} h" for p in result['sorted']]
    lines += ['', 'OPTIMAL BOX SELECTION (0/1 KNAPSACK)', 'Selected: ' + ', '.join(p['id'] for p in result['selected']), f"Total weight: {sum(p['weight'] for p in result['selected'])} kg", f"Total value score: {result['score']}", '', 'DELIVERY ROUTE (NEAREST NEIGHBOR HEURISTIC)', ' -> '.join(result['route']), f"Total distance: {result['distance']:.2f} km", '']
    lines += [f'{a} -> {b}: {d:.2f} km via {" -> ".join(path)}' for a,b,d,path in result['legs']]
    lines += ['', 'RUNTIME (MILLISECONDS)'] + [f'{k}: {v:.4f}' for k,v in result['times'].items()]
    report = '\n'.join(lines); print(report); (outdir/'delivery_plan.txt').write_text(report, encoding='utf-8')
    rows = benchmarks(graph, outdir)
    print('\nBenchmark chart and CSV saved in:', outdir)
    print('Benchmark samples:', len(rows))

if __name__ == '__main__': main()

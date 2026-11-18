import unittest
from smartcourier import merge_sort, knapsack, dijkstra, nearest_neighbor_route, load_graph, load_parcels, plan, BASE

class TestSmartCourier(unittest.TestCase):
    def setUp(self):
        self.graph = load_graph(BASE/'data'/'roads.csv')
        self.parcels = load_parcels(BASE/'data'/'parcels.csv')
    def test_sort(self):
        self.assertEqual(merge_sort([3, 1, 2], lambda x:x), [1, 2, 3])
    def test_knapsack(self):
        ps = [{'weight': 2, 'priority': 1, 'deadline': 50}, {'weight': 3, 'priority': 2, 'deadline': 50}]
        chosen, value, _ = knapsack(ps, 3)
        self.assertEqual(value, 200)
        self.assertEqual(len(chosen), 1)
    def test_dijkstra(self):
        distances, _ = dijkstra(self.graph, 'Depot')
        self.assertEqual(distances['B'], 6)
    def test_route(self):
        route, length, legs = nearest_neighbor_route(self.graph, 'Depot', ['A', 'C'])
        self.assertEqual(route[0], 'Depot')
        self.assertEqual(route[-1], 'Depot')
        self.assertGreater(length, 0)
        self.assertEqual(len(legs), 3)
    def test_capacity(self):
        result = plan(self.parcels, self.graph, 15)
        self.assertLessEqual(sum(p['weight'] for p in result['selected']), 15)
    def test_empty(self):
        result = plan([], self.graph, 15)
        self.assertEqual(result['route'], ['Depot', 'Depot'])
        self.assertEqual(result['distance'], 0)
    def test_negative_capacity(self):
        with self.assertRaises(ValueError): knapsack([], -1)

if __name__ == '__main__': unittest.main()

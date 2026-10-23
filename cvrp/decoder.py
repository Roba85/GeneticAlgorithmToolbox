from __future__ import annotations

from typing import Callable

__all__ = [
    "Tour",
    "SplitFn",
    "greedy_split",
    "route_cost",
    "routes_to_tour",
    "SPLIT_STRATEGIES",
]

# Customer ids, depot excluded.
Tour = list[int]

# (tour, demands, capacity, dist, depot) -> routes
SplitFn = Callable[[Tour, list[int], int, list[list[float]], int], list[list[int]]]


def greedy_split(
    tour: Tour,
    demands: list[int],
    capacity: int,
    dist: list[list[float]],
    depot: int = 0,
) -> list[list[int]]:
    """Group the tour into routes"""
    routes: list[list[int]] = []
    current: list[int] = []
    load = 0

    for node in tour:
        demand = demands[node]
        if demand > capacity:
            raise ValueError(
                f"customer {node} demands {demand}, more than a truck can carry ({capacity})"
            )

        if load + demand > capacity:
            routes.append([depot, *current, depot])
            current = []
            load = 0

        current.append(node)
        load += demand

    if current:
        routes.append([depot, *current, depot])

    return routes


def route_cost(route: list[int], dist: list[list[float]]) -> float:
    """Sum of all trips in a route"""
    total = 0
    for from_node, to_node in zip(route, route[1:]):
        total += dist[from_node][to_node]
    return total


def routes_to_tour(routes: list[list[int]]) -> Tour:
    """flatten routes into a single list"""
    tour: Tour = []
    for route in routes:
        if len(route) < 2:
            continue
        tour.extend(route[1:-1])  # remove first and last elements (zeros)
    return tour


SPLIT_STRATEGIES: dict[str, SplitFn] = {"greedy": greedy_split}

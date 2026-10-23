"""CVRP problem definition: chromosome, decoding and fitness."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ga_toolbox.representations import PermutationRepresentation
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ga_toolbox"))
    from representations import PermutationRepresentation

from cvrp.decoder import SplitFn, greedy_split, route_cost
from cvrp.parser import CVRPInstance

__all__ = ["CVRPProblem"]


class CVRPProblem:
    """Fitness and feasibility for one CVRP instance.

    Genes are customer indices: gene ``i`` is node ``i + 1``, since node 0 is
    the depot. A chromosome is therefore a permutation of ``0..n_nodes-2``.
    """

    CAPACITY_PENALTY = 1_000_000

    def __init__(
        self,
        instance: CVRPInstance,
        split_fn: SplitFn = greedy_split,
        rng=None,
    ) -> None:
        self.instance = instance
        self.split_fn = split_fn
        self.representation = PermutationRepresentation(instance.n_nodes - 1, rng)
        # larger than the worst possible feasible tour (2 * K * max_distance)
        self.extra_route_penalty = 10 * instance.n_nodes * instance.max_distance

    def decode(self, genes: list[int]) -> list[list[int]]:
        tour = []
        for gene in genes:
            tour.append(gene + 1)  # gene i -> node i + 1

        return self.split_fn(
            tour,
            self.instance.demands,
            self.instance.capacity,
            self.instance.distance_matrix,
            self.instance.depot,
        )

    def evaluate(self, genes: list[int]) -> tuple[float, list[list[int]]]:
        routes = self.decode(genes)
        return self.evaluate_routes(routes), routes

    def evaluate_routes(self, routes: list[list[int]]) -> float:
        cost = 0
        overload = 0
        for route in routes:
            cost += route_cost(route, self.instance.distance_matrix)
            load = 0
            for node in route[1:-1]:
                load += self.instance.demands[node]
            if load > self.instance.capacity:
                overload += load - self.instance.capacity

        extra_routes = 0
        if len(routes) > self.instance.num_vehicles:
            extra_routes = len(routes) - self.instance.num_vehicles

        return (
            cost + extra_routes * self.extra_route_penalty + overload * self.CAPACITY_PENALTY
        )

    def fitness_fn(self, genes: list[int]) -> float:
        return self.evaluate_routes(self.decode(genes))

    def feasible_fn(self, genes: list[int]) -> bool:
        return self.representation.is_valid(genes)

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
from pathlib import Path

try:
    from ga_toolbox.crossover import OrderCrossover
    from ga_toolbox.engine import GAConfig, GeneticAlgorithm
    from ga_toolbox.mutation import InversionMutation, SwapMutation
    from ga_toolbox.replacement import ElitistReplacement, GenerationalReplacement
    from ga_toolbox.selection import RouletteWheelSelection, TournamentSelection
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ga_toolbox"))
    from crossover import OrderCrossover
    from engine import GAConfig, GeneticAlgorithm
    from mutation import InversionMutation, SwapMutation
    from replacement import ElitistReplacement, GenerationalReplacement
    from selection import RouletteWheelSelection, TournamentSelection

from cvrp.decoder import route_cost
from cvrp.parser import CVRPInstance, load_example, load_json
from cvrp.problem import CVRPProblem

__all__ = ["build_ga", "format_solution", "main"]


def build_ga(
    instance: CVRPInstance,
    args: argparse.Namespace,
) -> tuple[GeneticAlgorithm, CVRPProblem]:
    """Create the problem and the configured GA."""
    problem = CVRPProblem(instance, rng=random.Random(args.seed))
    rng = random.Random(args.seed)

    if args.selection == "roulette":
        selection = RouletteWheelSelection(rng=rng)
    else:
        selection = TournamentSelection(rng=rng)

    crossover = OrderCrossover(rate=args.cx_rate, rng=rng)

    if args.mutation == "swap":
        mutation = SwapMutation(rate=args.mut_rate, rng=rng)
    else:
        mutation = InversionMutation(rate=args.mut_rate, rng=rng)

    if args.replacement == "generational":
        replacement = GenerationalReplacement()
    else:
        replacement = ElitistReplacement()

    config = GAConfig(
        pop_size=args.pop_size,
        generations=args.generations,
        maximize=False,
        stop_after_x_static_avg=args.patience,
        k_group_tournament=args.tournament_size,
        log_every=args.log_every,
        log_verbose=False,
    )

    ga = GeneticAlgorithm(
        config=config,
        representation=problem.representation,
        replacement=replacement,
        selection_method=selection,
        crossover_method=crossover,
        mutation_method=mutation,
        feasibility=problem.feasible_fn,
        fitness=problem.fitness_fn,
    )

    return ga, problem


def format_solution(routes: list[list[int]], instance: CVRPInstance) -> str:
    blocks = []
    total_distance = 0
    total_load = 0

    for index, route in enumerate(routes):
        route_distance = route_cost(route, instance.distance_matrix)
        route_load = 0

        block = f"Route for vehicle {index}:\n"
        for node in route[:-1]:
            route_load += instance.demands[node]
            block += f" {node} Load({route_load}) -> "

        block += f" {route[-1]} Load({route_load})\n"
        block += f"Distance of the route: {route_distance}m\n"
        block += f"Load of the route: {route_load}"

        blocks.append(block)
        total_distance += route_distance
        total_load += route_load

    output = f"Objective: {total_distance}\n"
    output += "\n\n".join(blocks)
    output += "\n\n"
    output += f"Total Distance of all routes: {total_distance}m\n"
    output += f"Total Load of all routes: {total_load}"
    return output


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the CVRP with the GA toolbox.")
    parser.add_argument(
        "--instance",
        type=Path,
        default=None,
        help="JSON instance (default: example mentioned in the assignment)",
    )
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection", choices=["tournament", "roulette"], default="tournament"
    )
    parser.add_argument(
        "--mutation", choices=["inversion", "swap"], default="inversion"
    )
    parser.add_argument(
        "--replacement", choices=["elitist", "generational"], default="elitist"
    )
    parser.add_argument("--cx-rate", type=float, default=0.9)
    parser.add_argument("--mut-rate", type=float, default=0.3)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="stop after this many generations without improvement",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true", help="print only the solution")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.instance is None:
        instance = load_example()
    else:
        instance = load_json(args.instance)

    if args.pop_size % 2 != 0:
        print("pop-size must be even", file=sys.stderr)
        return 2

    ga, problem = build_ga(instance, args)

    if args.quiet:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            result = ga.run_GA()
    else:
        result = ga.run_GA()

    if result is None:
        print("GA failed to build a feasible initial population", file=sys.stderr)
        return 1

    best_chromosome, _ = result
    _, routes = problem.evaluate(best_chromosome)

    if len(routes) > instance.num_vehicles:
        print(f"warning: best solution uses {len(routes)} vehicles", file=sys.stderr)

    print(format_solution(routes, instance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

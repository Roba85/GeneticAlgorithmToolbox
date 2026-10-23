from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["CVRPInstance", "EXAMPLE", "load_example", "load_json"]


@dataclass(frozen=True)
class CVRPInstance:
    """A validated CVRP instance. Node 0 is the depot."""

    num_vehicles: int
    capacity: int
    demands: list[int]
    distance_matrix: list[list[float]]
    depot: int = 0

    @property
    def n_nodes(self) -> int:
        return len(self.demands)

    @property
    def customers(self) -> list[int]:
        customers = []
        for node in range(self.n_nodes):
            if node != self.depot:
                customers.append(node)
        return customers

    @property
    def total_demand(self) -> int:
        return sum(self.demands)

    @property
    def max_distance(self) -> float:
        """Largest arc cost, used to size the penalties."""
        largest = self.distance_matrix[0][0]
        for row in self.distance_matrix:
            for value in row:
                if value > largest:
                    largest = value
        return largest


# OR-Tools CVRP example used by the assignment (17 nodes, 4 vehicles)
EXAMPLE: dict[str, Any] = {
    "num_vehicles": 4,
    "capacity": 15,
    "depot": 0,
    "demands": [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8],
    "distance_matrix": [
        [
            0,
            548,
            776,
            696,
            582,
            274,
            502,
            194,
            308,
            194,
            536,
            502,
            388,
            354,
            468,
            776,
            662,
        ],
        [
            548,
            0,
            684,
            308,
            194,
            502,
            730,
            354,
            696,
            742,
            1084,
            594,
            480,
            674,
            1016,
            868,
            1210,
        ],
        [
            776,
            684,
            0,
            992,
            878,
            502,
            274,
            810,
            468,
            742,
            400,
            1278,
            1164,
            1130,
            788,
            1552,
            754,
        ],
        [
            696,
            308,
            992,
            0,
            114,
            650,
            878,
            502,
            844,
            890,
            1232,
            514,
            628,
            822,
            1164,
            560,
            1358,
        ],
        [
            582,
            194,
            878,
            114,
            0,
            536,
            764,
            388,
            730,
            776,
            1118,
            400,
            514,
            708,
            1050,
            674,
            1244,
        ],
        [
            274,
            502,
            502,
            650,
            536,
            0,
            228,
            308,
            194,
            240,
            582,
            776,
            662,
            628,
            514,
            1050,
            708,
        ],
        [
            502,
            730,
            274,
            878,
            764,
            228,
            0,
            536,
            194,
            468,
            354,
            1004,
            890,
            856,
            514,
            1278,
            480,
        ],
        [
            194,
            354,
            810,
            502,
            388,
            308,
            536,
            0,
            342,
            388,
            730,
            468,
            354,
            320,
            662,
            742,
            856,
        ],
        [
            308,
            696,
            468,
            844,
            730,
            194,
            194,
            342,
            0,
            274,
            388,
            810,
            696,
            662,
            320,
            1084,
            514,
        ],
        [
            194,
            742,
            742,
            890,
            776,
            240,
            468,
            388,
            274,
            0,
            342,
            536,
            422,
            388,
            274,
            810,
            468,
        ],
        [
            536,
            1084,
            400,
            1232,
            1118,
            582,
            354,
            730,
            388,
            342,
            0,
            878,
            764,
            730,
            388,
            1152,
            354,
        ],
        [
            502,
            594,
            1278,
            514,
            400,
            776,
            1004,
            468,
            810,
            536,
            878,
            0,
            114,
            308,
            650,
            274,
            844,
        ],
        [
            388,
            480,
            1164,
            628,
            514,
            662,
            890,
            354,
            696,
            422,
            764,
            114,
            0,
            194,
            536,
            388,
            730,
        ],
        [
            354,
            674,
            1130,
            822,
            708,
            628,
            856,
            320,
            662,
            388,
            730,
            308,
            194,
            0,
            342,
            422,
            536,
        ],
        [
            468,
            1016,
            788,
            1164,
            1050,
            514,
            514,
            662,
            320,
            274,
            388,
            650,
            536,
            342,
            0,
            764,
            194,
        ],
        [
            776,
            868,
            1552,
            560,
            674,
            1050,
            1278,
            742,
            1084,
            810,
            1152,
            274,
            388,
            422,
            764,
            0,
            798,
        ],
        [
            662,
            1210,
            754,
            1358,
            1244,
            708,
            480,
            856,
            514,
            468,
            354,
            844,
            730,
            536,
            194,
            798,
            0,
        ],
    ],
}


# validation helpers
def _require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"instance is missing required key {key!r}")
    return data[key]


def _as_int(value: Any, field: str) -> int:
    # bool is an int subclass, so reject True/False explicitly
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def _as_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    return value


def _as_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list, got {type(value).__name__}")
    return list(value)


def _build_instance(data: Mapping[str, Any]) -> CVRPInstance:
    """Validate a raw mapping; errors name the offending field."""
    num_vehicles = _as_int(_require(data, "num_vehicles"), "num_vehicles")
    if num_vehicles < 1:
        raise ValueError(f"num_vehicles must be >= 1, got {num_vehicles}")

    capacity = _as_int(_require(data, "capacity"), "capacity")
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")

    depot = _as_int(data.get("depot", 0), "depot")
    if depot != 0:
        raise ValueError(
            f"depot must be 0 (the permutation encoding assumes it), got {depot}"
        )

    demands = []
    raw_demands = _as_list(_require(data, "demands"), "demands")
    for i, demand in enumerate(raw_demands):
        demands.append(_as_int(demand, f"demands[{i}]"))
    if len(demands) < 2:
        raise ValueError("demands must contain the depot and at least one customer")
    if demands[depot] != 0:
        raise ValueError(f"the depot must have demand 0, got {demands[depot]}")
    for node, demand in enumerate(demands):
        if demand < 0:
            raise ValueError(f"demands[{node}] must be >= 0, got {demand}")
        if demand > capacity:
            raise ValueError(
                f"demands[{node}] = {demand} exceeds the vehicle capacity {capacity}"
            )

    raw_matrix = _as_list(_require(data, "distance_matrix"), "distance_matrix")
    if len(raw_matrix) != len(demands):
        raise ValueError(
            f"distance_matrix has {len(raw_matrix)} rows but demands has {len(demands)} entries"
        )

    distance_matrix: list[list[float]] = []
    for i, raw_row in enumerate(raw_matrix):
        row = []
        raw_columns = _as_list(raw_row, f"distance_matrix[{i}]")
        for j, value in enumerate(raw_columns):
            row.append(_as_number(value, f"distance_matrix[{i}][{j}]"))
        if len(row) != len(demands):
            raise ValueError(
                f"distance_matrix[{i}] has {len(row)} columns, expected {len(demands)}"
            )
        if row[i] != 0:
            raise ValueError(f"distance_matrix[{i}][{i}] must be 0, got {row[i]}")
        distance_matrix.append(row)

    total_demand = sum(demands)
    total_capacity = num_vehicles * capacity
    if total_demand > total_capacity:
        raise ValueError(
            f"instance is infeasible: total demand {total_demand} exceeds "
            f"total capacity {num_vehicles} x {capacity} = {total_capacity}"
        )

    return CVRPInstance(
        num_vehicles=num_vehicles,
        capacity=capacity,
        demands=demands,
        distance_matrix=distance_matrix,
        depot=depot,
    )


# loaders
def load_example() -> CVRPInstance:
    return _build_instance(EXAMPLE)


def load_json(path: str | Path) -> CVRPInstance:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a JSON object, got {type(data).__name__}"
        )
    return _build_instance(data)

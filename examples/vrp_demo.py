# Copyright 2026 LIMEN Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VRP demo for LIMEN: 24 real NYC landmarks, 3 vehicles, solved via D-Wave's
local simulated annealer (free, no Leap credentials required).

Per project memory (limen/frontends/vrp.py's depot-duplication encoding),
gate-model QAOA never finds the true optimum on this constraint-heavy QUBO
shape even at small sizes, while D-Wave's annealer (real QPU or, as used
here, its free local SimulatedAnnealingSampler) finds it reliably. This
demo runs entirely offline/free — no AWS Braket, no D-Wave Leap account.

Usage::

    python examples/vrp_demo.py [--vehicles N] [--reads N]
"""

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from limen.frontends.vrp import decode_routes, distance_matrix, from_vrp
from limen.pipeline import _graph_qubo, run_pipeline

# ---------------------------------------------------------------------------
# 24 real NYC landmarks (lat, lon) — public, well-known coordinates.
# Index 0 (Empire State Building) is the depot.
# ---------------------------------------------------------------------------

LANDMARKS: list[tuple[str, float, float]] = [
    ("Empire State Building", 40.748817, -73.985428),
    ("Times Square", 40.758896, -73.985130),
    ("Central Park / Bethesda Fountain", 40.7794, -73.9694),
    ("Statue of Liberty", 40.6892, -74.0445),
    ("Brooklyn Bridge", 40.7061, -73.9969),
    ("One World Trade Center", 40.7127, -74.0134),
    ("Grand Central Terminal", 40.7527, -73.9772),
    ("Madison Square Garden", 40.7505, -73.9934),
    ("Rockefeller Center", 40.7587, -73.9787),
    ("Chrysler Building", 40.7516, -73.9755),
    ("Flatiron Building", 40.7411, -73.9897),
    ("Washington Square Park", 40.7308, -73.9973),
    ("Union Square", 40.7359, -73.9911),
    ("High Line (South End)", 40.7480, -74.0048),
    ("Lincoln Center", 40.7725, -73.9835),
    ("American Museum of Natural History", 40.7813, -73.9740),
    ("Metropolitan Museum of Art", 40.7794, -73.9632),
    ("Yankee Stadium", 40.8296, -73.9262),
    ("Citi Field", 40.7571, -73.8458),
    ("Brooklyn Museum", 40.6712, -73.9636),
    ("Coney Island / Luna Park", 40.5755, -73.9707),
    ("LaGuardia Airport", 40.7769, -73.8740),
    ("Prospect Park / Grand Army Plaza", 40.6602, -73.9690),
    ("Battery Park", 40.7033, -74.0170),
]

_EARTH_RADIUS_KM = 6371.0


def _project_equirectangular(
    coords: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Project (lat, lon) degrees to local planar (x, y) km via an
    equirectangular projection centred on the coordinate set's mean latitude.

    Adequate for a single-city scale (a few tens of km); not a substitute
    for a geodesic distance for larger regions.
    """
    lat0 = math.radians(sum(lat for lat, _ in coords) / len(coords))
    points = []
    for lat, lon in coords:
        x = math.radians(lon) * _EARTH_RADIUS_KM * math.cos(lat0)
        y = math.radians(lat) * _EARTH_RADIUS_KM
        points.append((x, y))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--reads", type=int, default=1000)
    args = parser.parse_args()

    names = [name for name, _, _ in LANDMARKS]
    latlon = [(lat, lon) for _, lat, lon in LANDMARKS]
    coords = _project_equirectangular(latlon)
    n_customers = len(coords) - 1  # depot excluded

    print("=== LIMEN VRP Demo: NYC landmarks ===")
    print(f"Depot       : {names[0]}")
    print(f"Customers   : {n_customers}")
    print(f"Vehicles    : {args.vehicles}")
    print(f"Backend     : dwave (local SimulatedAnnealingSampler, free)")
    print()

    print("[1/3] Building VRP QUBO ...")
    graph, customer_ids = from_vrp(coords, num_vehicles=args.vehicles, depot=0)
    qubo = _graph_qubo(graph)
    n_vars = len({v for pair in qubo for v in pair})
    print(f"      QUBO terms: {len(qubo)}, variables: {n_vars}")

    print(f"[2/3] Annealing ({args.reads} reads) ...")
    t0 = time.time()
    cert = run_pipeline(
        qubo,
        backend="dwave",
        dwave_num_reads=args.reads,
        encode_logical=False,
    )
    elapsed = time.time() - t0
    print(f"      Elapsed: {elapsed:.1f}s")
    print(f"      Energy: {cert.energy:.4f}")
    print(f"      Classical optimal: {cert.classical_energy}")
    print(f"      Is optimal: {cert.is_optimal}")
    print(f"      Success probability: {cert.success_probability * 100:.1f}%")

    print("[3/3] Decoding routes ...")
    routes = decode_routes(cert.solution, n_customers, args.vehicles, customer_ids)

    dist = distance_matrix(coords)
    result: dict = {
        "demo": "vrp_nyc_landmarks",
        "date": time.strftime("%Y-%m-%d"),
        "depot": names[0],
        "num_customers": n_customers,
        "num_vehicles": args.vehicles,
        "backend": "dwave (local SimulatedAnnealingSampler)",
        "reads": args.reads,
        "elapsed_seconds": elapsed,
        "energy": cert.energy,
        "classical_optimal_energy": cert.classical_energy,
        "is_optimal": cert.is_optimal,
        "success_probability": cert.success_probability,
        "feasible": routes is not None,
        "routes": None,
    }

    if routes is None:
        print("      Infeasible assignment (constraint violation).")
    else:
        route_details = []
        for k, route in enumerate(routes):
            stops = [names[0]] + [names[i] for i in route] + [names[0]]
            # Route distance: depot -> route[0] -> ... -> route[-1] -> depot
            full_path = [0] + route + [0]
            route_len = sum(
                dist[full_path[i]][full_path[i + 1]] for i in range(len(full_path) - 1)
            )
            print(f"      Vehicle {k}: {' -> '.join(stops)}")
            print(f"        Distance: {route_len:.2f} km")
            route_details.append(
                {
                    "vehicle": k,
                    "stops": stops,
                    "distance_km": route_len,
                }
            )
        result["routes"] = route_details
        result["total_distance_km"] = sum(r["distance_km"] for r in route_details)
        print(f"      Total fleet distance: {result['total_distance_km']:.2f} km")

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"vrp_nyc_{stamp}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print()
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()

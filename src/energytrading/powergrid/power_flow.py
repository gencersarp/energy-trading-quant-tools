"""DC power flow solver for energy trading grid modeling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Bus:
    bus_id: int
    name: str
    bus_type: str    # 'slack' | 'PQ' | 'PV'
    P_gen: float = 0.0   # MW injection
    P_load: float = 0.0  # MW load
    Q_load: float = 0.0
    voltage_pu: float = 1.0


@dataclass
class Branch:
    from_bus: int
    to_bus: int
    reactance: float   # per unit
    susceptance: float = 0.0
    capacity_mw: float = 1000.0
    status: bool = True

    def __post_init__(self):
        if self.susceptance == 0.0 and self.reactance != 0.0:
            self.susceptance = 1.0 / self.reactance


class DCPowerFlowSolver:
    """DC power flow solver: B * theta = P_injections."""

    def __init__(self):
        self._buses: Dict[int, Bus] = {}
        self._branches: List[Branch] = []
        self._slack_bus: Optional[int] = None

    def add_bus(self, bus: Bus) -> None:
        self._buses[bus.bus_id] = bus
        if bus.bus_type == "slack":
            self._slack_bus = bus.bus_id

    def add_branch(self, branch: Branch) -> None:
        if branch.status:
            self._branches.append(branch)

    def build_admittance_matrix(self) -> np.ndarray:
        """Build B matrix (imaginary part of Ybus for DC flow)."""
        ids = sorted(self._buses.keys())
        n = len(ids)
        idx_map = {bus_id: i for i, bus_id in enumerate(ids)}
        B = np.zeros((n, n))
        for br in self._branches:
            i, j = idx_map[br.from_bus], idx_map[br.to_bus]
            b = br.susceptance
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b
        return B

    def solve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve DC power flow. Returns (voltage_angles, branch_flows, total_loss)."""
        ids = sorted(self._buses.keys())
        n = len(ids)
        idx_map = {bus_id: i for i, bus_id in enumerate(ids)}

        # Net injections
        P = np.array([self._buses[bid].P_gen - self._buses[bid].P_load
                      for bid in ids])

        B = self.build_admittance_matrix()

        # Remove slack bus row/col
        slack_idx = idx_map.get(self._slack_bus, 0)
        mask = np.ones(n, dtype=bool)
        mask[slack_idx] = False
        B_red = B[np.ix_(mask, mask)]
        P_red = P[mask]

        try:
            theta_red = np.linalg.solve(B_red, P_red)
        except np.linalg.LinAlgError:
            theta_red = np.linalg.lstsq(B_red, P_red, rcond=None)[0]

        theta = np.zeros(n)
        theta[mask] = theta_red

        # Branch flows
        flows = {}
        for br in self._branches:
            i, j = idx_map[br.from_bus], idx_map[br.to_bus]
            flow = br.susceptance * (theta[i] - theta[j])
            flows[(br.from_bus, br.to_bus)] = flow

        # DC approximation: losses = 0
        return theta, flows, 0.0

    def check_congestion(self) -> List[Dict]:
        _, flows, _ = self.solve()
        congested = []
        for br in self._branches:
            flow = abs(flows.get((br.from_bus, br.to_bus), 0.0))
            if flow > br.capacity_mw * 0.99:
                congested.append({
                    "from": br.from_bus,
                    "to": br.to_bus,
                    "flow_mw": flow,
                    "capacity_mw": br.capacity_mw,
                    "loading_pct": flow / br.capacity_mw * 100,
                })
        return congested

    def n_minus_1_contingency(self) -> Dict[Tuple, List[Dict]]:
        """N-1 security analysis: check congestion for each branch outage."""
        original_branches = list(self._branches)
        results = {}
        for i, outage_branch in enumerate(original_branches):
            self._branches = [b for j, b in enumerate(original_branches) if j != i]
            congested = self.check_congestion()
            if congested:
                key = (outage_branch.from_bus, outage_branch.to_bus)
                results[key] = congested
        self._branches = original_branches
        return results


class GridTopology:
    """Network representation for grid analysis."""

    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.nodes = nodes
        self.edges = edges
        self._build_adjacency()

    def _build_adjacency(self):
        n = len(self.nodes)
        self.node_idx = {n["id"]: i for i, n in enumerate(self.nodes)}
        self.adj = np.zeros((n, n))
        for e in self.edges:
            i = self.node_idx.get(e["from"], -1)
            j = self.node_idx.get(e["to"], -1)
            if i >= 0 and j >= 0:
                self.adj[i, j] = 1
                self.adj[j, i] = 1

    @classmethod
    def from_dict(cls, nodes: List[Dict], edges: List[Dict]) -> "GridTopology":
        return cls(nodes, edges)

    def degree_centrality(self) -> Dict:
        n = len(self.nodes)
        degrees = self.adj.sum(axis=1)
        return {self.nodes[i]["id"]: float(degrees[i] / (n - 1))
                for i in range(n)}

    def betweenness_centrality(self) -> Dict:
        """Approximate betweenness via BFS shortest paths."""
        n = len(self.nodes)
        centrality = np.zeros(n)
        for s in range(n):
            # BFS from source s
            visited = np.full(n, False)
            dist = np.full(n, np.inf)
            sigma = np.zeros(n)
            pred = [[] for _ in range(n)]
            dist[s] = 0
            sigma[s] = 1
            queue = [s]
            visited[s] = True
            stack = []
            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in np.where(self.adj[v] > 0)[0]:
                    if dist[w] == np.inf:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            delta = np.zeros(n)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / (sigma[w] + 1e-8)) * (1 + delta[w])
                if w != s:
                    centrality[w] += delta[w]
        total = (n - 1) * (n - 2) / 2
        centrality /= (total + 1e-8)
        return {self.nodes[i]["id"]: float(centrality[i]) for i in range(n)}

    def islanding_risk(self) -> List[Dict]:
        """Identify nodes at risk of isolation (low degree)."""
        degrees = self.adj.sum(axis=1)
        return [{"id": self.nodes[i]["id"], "degree": int(degrees[i])}
                for i in range(len(self.nodes)) if degrees[i] <= 1]

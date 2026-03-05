"""Locational Marginal Price (LMP) calculator."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class LMPCalculator:
    """Compute LMPs from nodal power balance and network constraints."""

    def compute_lmp(self, nodal_demands: Dict[int, float],
                    nodal_supplies: Dict[int, float],
                    marginal_costs: Dict[int, float],
                    shift_factors: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Simplified LMP = energy + congestion + loss components.
        Without shift factors, returns uniform energy component.
        """
        # Energy component: marginal cost of cheapest unconstrained generator
        nodes = sorted(set(nodal_demands.keys()) | set(nodal_supplies.keys()))
        total_demand = sum(nodal_demands.values())
        total_supply = sum(nodal_supplies.values())

        # Energy price: merit order clearing
        if marginal_costs:
            sorted_gen = sorted(marginal_costs.items(), key=lambda x: x[1])
            energy_price = 0.0
            cum_supply = 0.0
            for bus_id, mc in sorted_gen:
                cum_supply += nodal_supplies.get(bus_id, 0.0)
                energy_price = mc
                if cum_supply >= total_demand:
                    break
        else:
            energy_price = 50.0  # default

        # Without network model: LMP = energy price for all nodes
        lmp = {}
        for node in nodes:
            # Simplified: uniform energy + small congestion noise
            cong = 0.0
            if shift_factors is not None and node < len(shift_factors):
                cong = float(shift_factors[node] * 2.0)  # mock
            lmp[node] = energy_price + cong
        return lmp

    def decompose_lmp(self, lmp: Dict[int, float],
                       energy_price: float) -> Dict[int, Dict[str, float]]:
        """Decompose LMP into energy, congestion, loss components."""
        decomposed = {}
        for node, price in lmp.items():
            congestion = price - energy_price
            decomposed[node] = {
                "energy": energy_price,
                "congestion": congestion,
                "loss": 0.0,  # DC approx: no losses
                "total": price,
            }
        return decomposed

    def congestion_rent(self, branch_flows: Dict[Tuple, float],
                         lmp_dict: Dict[int, float],
                         capacity: float) -> Dict[Tuple, float]:
        """Congestion rent = (LMP_to - LMP_from) * flow."""
        rents = {}
        for (from_bus, to_bus), flow in branch_flows.items():
            lmp_from = lmp_dict.get(from_bus, 0.0)
            lmp_to = lmp_dict.get(to_bus, 0.0)
            rents[(from_bus, to_bus)] = (lmp_to - lmp_from) * flow
        return rents

    def ftrs_value(self, ftr_rights: Dict[Tuple, float],
                    lmp_df: pd.DataFrame) -> pd.Series:
        """
        Financial Transmission Rights value.
        ftr_rights: {(from, to): MW_entitlement}
        """
        values = pd.Series(0.0, index=lmp_df.index)
        for (f, t), mw in ftr_rights.items():
            if str(f) in lmp_df.columns and str(t) in lmp_df.columns:
                values += (lmp_df[str(t)] - lmp_df[str(f)]) * mw
        return values


class NodalPricingAnalyzer:
    """Historical LMP analysis for price zone identification."""

    def historical_spread(self, lmp_df: pd.DataFrame,
                           node1: str, node2: str) -> pd.Series:
        """Price spread between two nodes over time."""
        return lmp_df[node1] - lmp_df[node2]

    def congestion_frequency(self, branch_name: str,
                              lmp_df: pd.DataFrame,
                              threshold: float = 1.0) -> float:
        """Fraction of hours a branch is congested (price diff > threshold)."""
        if branch_name not in lmp_df.columns:
            return 0.0
        return float((lmp_df[branch_name].abs() > threshold).mean())

    def price_zone_mapping(self, lmp_df: pd.DataFrame,
                            n_zones: int = 5) -> Dict[str, int]:
        """K-means clustering of nodes by price behaviour."""
        data = lmp_df.T.values
        km = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
        labels = km.fit_predict(data)
        return {node: int(label) for node, label in zip(lmp_df.columns, labels)}

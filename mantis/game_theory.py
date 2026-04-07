"""Game theory algorithms for pricing decisions."""
from __future__ import annotations

import logging
import math
from collections import defaultdict

from .types import Orderbook

logger = logging.getLogger("mantis.game_theory")


# ── Markov Price Drift ──

def markov_price_drift(recent_prices: list[float], bins: int = 5) -> float:
    """Estimate price drift using discrete Markov transition matrix.

    Discretizes prices into bins, builds transition matrix,
    returns expected drift from current state.
    """
    if len(recent_prices) < 3:
        return 0.0

    # Discretize into bins
    min_p = min(recent_prices)
    max_p = max(recent_prices)
    rng = max_p - min_p
    if rng < 0.001:
        return 0.0  # price is flat

    bin_size = rng / bins

    def to_bin(p: float) -> int:
        b = int((p - min_p) / bin_size)
        return min(b, bins - 1)

    def bin_center(b: int) -> float:
        return min_p + (b + 0.5) * bin_size

    # Build transition counts
    transitions: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(recent_prices) - 1):
        s_from = to_bin(recent_prices[i])
        s_to = to_bin(recent_prices[i + 1])
        transitions[s_from][s_to] += 1

    # Current state
    current_bin = to_bin(recent_prices[-1])
    current_center = bin_center(current_bin)

    # Expected next price
    row = transitions.get(current_bin, {})
    total = sum(row.values())
    if total == 0:
        return 0.0

    expected = sum(bin_center(b) * count / total for b, count in row.items())
    drift = expected - current_center

    return drift


# ── Glosten-Milgrom Minimum Spread ──

def glosten_milgrom_min_spread(
    recent_fills: list[dict],
    informed_ratio: float = 0.15,
    expected_jump: float = 0.05,
) -> float:
    """Compute minimum safe spread using Glosten-Milgrom model.

    min_spread >= mu * expected_instant_price_jump

    Where mu = estimated fraction of informed traders.
    If we detect more adverse selection, increase the estimate.
    """
    if not recent_fills:
        return informed_ratio * expected_jump

    # Detect adverse selection: large orders that moved the price
    adverse_count = 0
    for fill in recent_fills:
        size = fill.get("size", 0)
        if size > 100:  # "large" order threshold
            adverse_count += 1

    # Adjust informed ratio based on observed adverse selection
    adverse_rate = adverse_count / len(recent_fills) if recent_fills else 0
    adjusted_mu = min(informed_ratio + adverse_rate * 0.5, 0.5)

    min_spread = adjusted_mu * expected_jump
    return min_spread


# ── Nash Optimal Spread ──

def nash_optimal_spread(
    orderbook: Orderbook,
    min_spread: float,
    aggression: float = 0.9,
) -> float:
    """Determine spread based on competitor analysis (Nash/Bertrand).

    Look at existing orderbook to see what competitors offer.
    Set our spread slightly tighter (aggression < 1) but never below min_spread.
    """
    if not orderbook.bids or not orderbook.asks:
        return max(min_spread, 0.01)

    # Competitor spread = current best bid-ask spread
    competitor_spread = orderbook.best_ask - orderbook.best_bid

    # We want to be slightly tighter than competitors
    our_spread = competitor_spread * aggression

    # But never below the GM safety floor
    our_spread = max(our_spread, min_spread)

    # And never below 0.2 cents (gas costs make sub-cent impractical)
    our_spread = max(our_spread, 0.002)

    return our_spread


# ── CFR (Counterfactual Regret Minimization) ──

class CFREngine:
    """Simple CFR for spread strategy selection.

    Strategies: different half-spread values.
    Utility: daily PnL from using that spread.
    """

    def __init__(self, strategies: list[float]):
        self.strategies = strategies
        self.n = len(strategies)
        self.cumulative_regret = [0.0] * self.n
        self.cumulative_strategy = [0.0] * self.n
        self.rounds = 0

    def get_strategy(self) -> list[float]:
        """Get current mixed strategy (probability distribution over spreads)."""
        # Regret-matching: positive regrets determine probabilities
        positive_regret = [max(0, r) for r in self.cumulative_regret]
        total = sum(positive_regret)

        if total > 0:
            return [r / total for r in positive_regret]
        else:
            # Uniform if no positive regret
            return [1.0 / self.n] * self.n

    def select_spread(self) -> float:
        """Select a half-spread based on current strategy distribution."""
        import random
        probs = self.get_strategy()
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return self.strategies[i]
        return self.strategies[-1]

    def update(self, utilities: list[float]):
        """Update regrets given utility (PnL) for each strategy.

        utilities[i] = what we would have earned using strategy i.
        """
        strategy = self.get_strategy()
        expected_utility = sum(s * u for s, u in zip(strategy, utilities))

        for i in range(self.n):
            regret = utilities[i] - expected_utility
            self.cumulative_regret[i] += regret
            self.cumulative_strategy[i] += strategy[i]

        self.rounds += 1

    def get_average_strategy(self) -> list[float]:
        """Get the time-averaged strategy (converges to Nash equilibrium)."""
        total = sum(self.cumulative_strategy)
        if total > 0:
            return [s / total for s in self.cumulative_strategy]
        return [1.0 / self.n] * self.n

    def load_state(self, state: list[dict]):
        """Load CFR state from database."""
        for row in state:
            key = row["strategy_key"]
            try:
                idx = self.strategies.index(float(key))
                self.cumulative_regret[idx] = row["cumulative_regret"]
                self.cumulative_strategy[idx] = row["cumulative_strategy"]
                self.rounds = max(self.rounds, row["rounds"])
            except (ValueError, IndexError):
                continue

    def save_state(self) -> list[tuple[str, float, float, int]]:
        """Return state tuples for database persistence."""
        return [
            (str(s), self.cumulative_regret[i], self.cumulative_strategy[i], self.rounds)
            for i, s in enumerate(self.strategies)
        ]


# ── Evolutionary Game Theory: Market Ecosystem Detection ──

def detect_ecosystem_state(
    orderbook: Orderbook,
    prev_orderbook: Orderbook | None,
) -> str:
    """Detect market ecosystem state changes.

    Returns: 'normal', 'crowded', 'dead', 'whale_entered'
    """
    if not orderbook.bids or not orderbook.asks:
        return "dead"

    current_depth = sum(l.size for l in orderbook.bids) + sum(l.size for l in orderbook.asks)

    if current_depth < 100:
        return "dead"

    if prev_orderbook and prev_orderbook.bids and prev_orderbook.asks:
        prev_depth = sum(l.size for l in prev_orderbook.bids) + sum(l.size for l in prev_orderbook.asks)

        if prev_depth > 0:
            ratio = current_depth / prev_depth
            if ratio > 3.0:
                return "whale_entered"

    # Check if spread is too tight (price war)
    if orderbook.spread_cents < 0.3:
        return "crowded"

    return "normal"

from typing import Callable

def compute_delta(pricer: Callable[[float], float], S: float, bump: float = 1e-4) -> float:
    """Numerical delta via central difference."""
    return (pricer(S + bump) - pricer(S - bump)) / (2 * bump)

def compute_gamma(pricer: Callable[[float], float], S: float, bump: float = 1e-4) -> float:
    """Numerical gamma via central difference."""
    return (pricer(S + bump) - 2 * pricer(S) + pricer(S - bump)) / (bump ** 2)
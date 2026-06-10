# Copyright 2025 LIMEN Contributors. Apache 2.0.
"""Pure-Python fallback for StackelbergSolver when limen_core is not built."""

from dataclasses import dataclass


@dataclass
class EquilibriumScore:
    kappa: float
    confidence: float
    energy_gap: float
    iterations: int
    kappa_std: float = 0.0

    def __repr__(self) -> str:
        return (
            f"EquilibriumScore(kappa={self.kappa:.4f}, "
            f"confidence={self.confidence:.4f}, "
            f"energy_gap={self.energy_gap:.4f}, "
            f"kappa_std={self.kappa_std:.4f})"
        )


class StackelbergSolver:
    def __init__(
        self,
        target_kappa: float = 0.85,
        max_iterations: int = 50,
        learning_rate: float = 0.1,
    ) -> None:
        self.target_kappa = target_kappa
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def solve(
        self,
        confidences: list,
        best_energies: list,
        second_best_energies: list,
        chain_break_fractions: list,
        current_chain_strength: float,
    ) -> tuple:
        if not confidences:
            score = EquilibriumScore(
                kappa=0.0, confidence=0.0, energy_gap=0.0, iterations=0
            )
            return current_chain_strength, score

        scores = []
        kappas = []
        for i in range(len(confidences)):
            conf = confidences[i]
            best_e = best_energies[i]
            second_e = second_best_energies[i]
            cbf = chain_break_fractions[i]
            gap_term = min(abs(second_e - best_e), 10.0) / 10.0
            cbf_penalty = 1.0 - max(0.0, min(1.0, cbf))
            kappa = 0.5 * conf + 0.3 * gap_term + 0.2 * cbf_penalty
            kappa = max(0.0, min(1.0, kappa))
            kappas.append(kappa)
            scores.append((kappa, conf, abs(second_e - best_e), i))

        # kappa_std
        if len(kappas) >= 2:
            mean = sum(kappas) / len(kappas)
            variance = sum((k - mean) ** 2 for k in kappas) / len(kappas)
            kappa_std = variance ** 0.5
        else:
            kappa_std = 0.0

        stability_penalty = min(0.9, kappa_std * 5.0)
        effective_lr = self.learning_rate * (1.0 - stability_penalty)

        best = max(scores, key=lambda x: x[0])
        best_kappa, best_conf, best_gap, best_iter = best

        score = EquilibriumScore(
            kappa=best_kappa,
            confidence=best_conf,
            energy_gap=best_gap,
            iterations=best_iter,
            kappa_std=kappa_std,
        )

        if best_kappa >= self.target_kappa:
            return current_chain_strength, score

        adjustment = effective_lr * (1.0 - best_kappa)
        new_cs = current_chain_strength * (1.0 + adjustment)
        return new_cs, score

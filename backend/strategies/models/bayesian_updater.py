import numpy as np

class BayesianUpdater:
    """
    Bayesian Updater for adjusting beliefs based on new evidence.
    Use for strategy confidence scoring, macro regime inference, recession probability, etc.
    """

    def __init__(self, prior: float):
        """
        Initialize with a prior probability (between 0 and 1).
        Example: prior = 0.5 means a 50% belief before observing new evidence.
        """
        if not 0 <= prior <= 1:
            raise ValueError("Prior must be between 0 and 1.")
        self.prior = prior

    def update(self, likelihood: float, evidence_prob: float) -> float:
        """
        Performs Bayesian update.

        Parameters:
        - likelihood (P(E|H)): Probability of seeing evidence if hypothesis is true
        - evidence_prob (P(E)): Probability of seeing evidence under all conditions

        Returns:
        - posterior (P(H|E)): Updated probability of hypothesis given evidence
        """
        if not 0 <= likelihood <= 1 or not 0 <= evidence_prob <= 1:
            raise ValueError("Likelihood and evidence_prob must be between 0 and 1.")

        numerator = likelihood * self.prior
        denominator = evidence_prob

        if denominator == 0:
            return self.prior  # avoid division by zero

        posterior = numerator / denominator
        self.prior = posterior
        return posterior

    def batch_update(self, updates: list) -> float:
        """
        Perform multiple Bayesian updates in sequence.

        Each update in `updates` is a tuple: (likelihood, evidence_prob)

        Returns:
        - Final posterior probability after all updates
        """
        for likelihood, evidence_prob in updates:
            self.update(likelihood, evidence_prob)
        return self.prior

# Example Usage:
if __name__ == "__main__":
    updater = BayesianUpdater(prior=0.4)
    print("Initial belief:", updater.prior)

    # Example: Evidence supports hypothesis with 70% chance, and occurs 60% of the time
    posterior = updater.update(likelihood=0.7, evidence_prob=0.6)
    print("Posterior after 1 update:", posterior)

    # Batch updates
    updates = [(0.8, 0.7), (0.6, 0.5)]
    final = updater.batch_update(updates)
    print("Final posterior after batch updates:", final)
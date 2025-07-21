import math

class EdgeDecay:
    def __init__(self, decay_days: int, final_weight: float):
        self.decay_days = decay_days
        self.final_weight = final_weight

        # Precompute constants
        self.linear_slope = (final_weight - 1.0) / decay_days
        self.exp_rate = math.log(final_weight) / decay_days
        self.log_k = (1.0 - final_weight) / math.log1p(decay_days)
        self.sigmoid_midpoint = decay_days / 2
        self.sigmoid_steepness = 10 / decay_days  # you can tune this if desired

    def linear(self, days: int) -> float:
        if days >= self.decay_days:
            return 0.0
        w = 1.0 + self.linear_slope * days
        return max(self.final_weight, w)

    def exponential(self, days: int) -> float:
        if days >= self.decay_days:
            return 0.0
        w = math.exp(self.exp_rate * days)
        return max(self.final_weight, w)

    def logarithmic(self, days: int) -> float:
        if days >= self.decay_days:
            return 0.0
        w = 1.0 - self.log_k * math.log1p(days)
        return max(self.final_weight, w)

    def sigmoid(self, days: int) -> float:
        if days >= self.decay_days:
            return 0.0
        w = 1.0 - (1.0 - self.final_weight) / (1 + math.exp(-self.sigmoid_steepness * (days - self.sigmoid_midpoint)))
        return max(self.final_weight, w)
    
    def quadratic(self, days: int) -> float:
        if days >= self.decay_days:
            return 0.0
        frac = days / self.decay_days
        w = 1.0 - (frac ** 2) * (1.0 - self.final_weight)
        return max(self.final_weight, w)

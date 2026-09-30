class Regress:
    def __init__(self, ypred: list, yactual: list):
        if len(ypred) != len(yactual):
            raise ValueError("ypred and yactual must be the same length!")
            
        self.ypred = ypred
        self.yactual = yactual
        self.n = len(yactual)

    def mse(self):
        return sum((act - pred) ** 2 for act, pred in zip(self.yactual, self.ypred)) / self.n

    def mae(self):
        """Calculates Mean Absolute Error (Linear scale error)."""
        return sum(abs(act - pred) for act, pred in zip(self.yactual, self.ypred)) / self.n

    def r2_score(self):
        mean_actual = sum(self.yactual) / self.n
        ss_res = sum((act - pred) ** 2 for act, pred in zip(self.yactual, self.ypred))
        ss_tot = sum((act - mean_actual) ** 2 for act in self.yactual)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    def adjusted_r2(self, num_features: int):
        """Calculates Adjusted R-squared to punish overfitting."""
        r2 = self.r2_score()
        if self.n - num_features - 1 <= 0:
            raise ValueError("Too many features for the number of data points sample size!")
            
        numerator = (1 - r2) * (self.n - 1)
        denominator = self.n - num_features - 1
        return 1 - (numerator / denominator)

class Classify:
    def __init__(self, ypred: list, yactual: list, pos_label=None):
        if len(ypred) != len(yactual):
            raise ValueError("ypred and yactual must be the same length!")
            
        self.ypred = ypred
        self.yactual = yactual
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        unique_vals = set(ypred + yactual)
        if len(unique_vals) != 2:
            raise ValueError(f"Data must contain exactly two distinct classes. Found: {unique_vals}")
        
        unique_list = list(unique_vals)
        if pos_label is not None:
            if pos_label not in unique_vals:
                raise ValueError(f"Selected pos_label '{pos_label}' not found in data.")
            self.p = pos_label
            unique_list.remove(pos_label)
            self.n = unique_list[0]
        else:
            self.p, self.n = unique_list[0], unique_list[1]
            
        self._initialize_predictions()

    def _initialize_predictions(self):
        for pred, actual in zip(self.ypred, self.yactual):
            if pred == actual and pred == self.p:
                self.tp += 1
            elif pred == actual and pred == self.n:
                self.tn += 1
            elif pred != actual and pred == self.p:
                self.fp += 1
            elif pred != actual and pred == self.n:
                self.fn += 1

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def recall(self):
        denom = self.tp + self.fn  
        return self.tp / denom if denom > 0 else 0.0          

    def F1score(self):
        p = self.precision()
        r = self.recall()
        denom = p + r
        return 2 * (p * r) / denom if denom > 0 else 0.0

    def display_confusion_matrix(self):
        """Prints a scannable text-based Confusion Matrix."""
        # Calculate maximum label length for clean padding
        p_str, n_str = str(self.p), str(self.n)
        width = max(len(p_str), len(n_str), 10)
        
        print("\n" + "="*45)
        print("              CONFUSION MATRIX")
        print("="*45)
        print(f"{'':<{width}} | {'PRED ' + p_str:<{width}} | {'PRED ' + n_str:<{width}}")
        print("-" * (width * 3 + 7))
        print(f"{'ACTUAL ' + p_str:<{width}} | {self.tp:<{width}} | {self.fn:<{width}}")
        print(f"{'ACTUAL ' + n_str:<{width}} | {self.fp:<{width}} | {self.tn:<{width}}")
        print("="*45 + "\n")
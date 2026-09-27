class Evaluate:
    def __init__(self,y,y_pred):
        self.y=y
        self.y_pred=y_pred
    def get_accuracy(self):
        correct = sum(1 for t, p in zip(self.y, self.y_pred) if t == p)
        return correct / len(self.y)
    def get_confusion_matrix(y_true, y_pred):
        tp, tn, fp, fn = 0, 0, 0, 0
        for t, p in zip(y_true, y_pred):
            if t == 1 and p == 1: tp += 1
            elif t == 0 and p == 0: tn += 1
            elif t == 0 and p == 1: fp += 1
            elif t == 1 and p == 0: fn += 1
            
        print(f"True Negatives (TN): {tn}  | False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}  | True Positives (TP):  {tp}")
        return tp, tn, fp, fn


class Logistics_regression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = None
        self.b = None
        self.len = len(self.X)
        
    def gradient(self, m, b):
        sum_m = 0
        sum_b = 0
        for i in range(self.len):
            z = m * self.X[i] + b
           
            p = 1 / (1 + (1 / (2.71828 ** z)) if z >= 0 else 1 + (2.71828 ** -z))
            
            sum_m += (p - self.y[i]) * self.X[i]
            sum_b += (p - self.y[i])
        return sum_m / self.len, sum_b / self.len
        
    def fit(self, epoch=100000, lr=0.01):
        m = 0
        b = 0
        if self.m is not None:
            m = self.m 
            b = self.b
        for epoches in range(epoch):
            grad_m, grad_b = self.gradient(m, b)
            m -= grad_m * lr
            b -= grad_b * lr
            if epoches % 100 == 0:
                print(f"Epoch:{epoches}")
        self.m = m
        self.b = b
        
    def predict(self, X, threshold=0.5):
        if self.m is None or self.b is None:
            self.fit()
            

        z = self.m * X + self.b
        probability = 1 / (1 + (1 / (2.71828 ** z)) if z >= 0 else 1 + (2.71828 ** -z))
        
        return 1 if probability >= threshold else 0

logi = Logistics_regression( [
    30, 32, 35, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    55, 56, 57, 58, 59, 60, 60, 61, 62, 63, 64, 65, 65, 66, 67, 68, 69, 70, 70, 71,
    72, 73, 74, 75, 75, 76, 77, 78, 79, 80, 80, 81, 82, 83, 84, 85, 85, 86, 87, 88,
    89, 90, 90, 91, 92, 93, 94, 95, 95, 96, 97, 98, 34, 43, 53, 62, 72, 81, 91, 39,
    48, 58, 67, 77, 86, 96, 36, 46, 56, 65, 75, 84, 94, 33, 44, 54, 63, 73, 83, 93
],  [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1
])
print("Final Prediction:", logi.predict(90))

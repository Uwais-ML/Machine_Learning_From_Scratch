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


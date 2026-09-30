from Methods.classic_methods import Methods
from Eval.Regression import Regress

class Linear_regression:
    def __init__(self,X,y,Normalized=None):  #initialized_varaibles
        self.X=X
        self.y=y
        self.len=len(X)
        self.Normalized=Normalized
        self.method=Methods(self.X,self.y)
        self.m=None
        self.b=None
        if not Normalized is None:
            self.X,self.y,self.max,self.min,self.x_max,self.x_min=self.method.Normalization()
    def _mse(self, m, b):
        return sum((self.y[i] - (m * self.X[i] + b)) ** 2 for i in range(self.len)) / self.len
    def gradient(self,m,b):
        sum_m=0
        sum_b=0
        for i in range(self.len):
            prediction = m * self.X[i] + b
            error = self.y[i] - prediction
            sum_m += self.X[i] * error
            sum_b += error

       
        grad_m = (-2 / self.len) * sum_m
        grad_b = (-2 / self.len) * sum_b
        return grad_m,grad_b
    def fit(self,lr=0.005,epoches=100000):
        m = 0.0 if self.m is None else self.m
        b = 0.0 if self.b is None else self.b
        for i in range(epoches):
            gradient_m,gradient_b=self.gradient(m,b)
            m=m-gradient_m*lr
            b=b-gradient_b*lr
            error=self._mse(m,b)
            to_stop=self.method.Early_stopping(error)
            if i % 1000 == 0:
                        print(f"Epoch {i}: m = {m:.4f}, b = {b:.4f}")
            if to_stop:
                self.method.save_checkpoint(m,b)
                print("Save the checkpoint")
                self.m=m
                self.b=b
                return m,b
        self.m=m
        self.b=b
        return m,b
    def predict(self,X):
        if self.m is None:
            self.fit()
        m, b = self.m, self.b
        if not self.Normalized is None:
            X = (X - self.x_min) / (self.x_max - self.x_min)  
        y= m*X + b
        if not self.Normalized is None:
            y=y*(self.max-self.min)+self.min
            print(y)
        return y
    def Eval(self):
        if self.m is None:
            self.fit()
        y_pred=[self.m*x + self.b for x in self.X]
        eval_metric=Regress(ypred=y_pred,yactual=self.y)
        print("R2 Score:", eval_metric.r2_score())
        print("MSE:", eval_metric.mse())     


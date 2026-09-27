class Methods:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.len=len(y)
        self.checkpoint=0
    def Normalization(self):
        x_norm=[]
        y_norm=[]
        x_min=min(self.X)
        x_max=max(self.X)
        y_min=min(self.y)
        y_max=max(self.y)
        for i in range(self.len):
            x_norm.append((self.X[i]-x_min)/(x_max-x_min))
            y_norm.append((self.y[i]-y_min)/(y_max-y_min))
        return x_norm,y_norm,y_max,y_min,x_max,x_min
    def save_checkpoint(self,m,b):
        combined_string=f"{m},{b}"
        with open(f"Linear_model_{self.checkpoint}","w") as f:
            f.write(combined_string)
        self.checkpoint+=1
    def Early_stopping(self,error,stopping_error=1e-7):
        return error<stopping_error
class Evaluation:
    def __init__(self,y_predicted,y_real):
        self.y_pred=y_predicted
        self.y_real=y_real
    def R_Square(self):
        mean = sum(self.y_real) / len(self.y_real)
        error=0
        for i in range(len(self.y_pred)):
            error+=(((self.y_pred[i]-self.y_real[i])**2)/((self.y_pred[i]-mean)**2))
        print("The R2 ERROR IS :",1-error)
    def _mse(self, m, b):
        mse = sum((self.y_real[i] - (self.y_pred[i])) ** 2 for i in range(len(self.y_real))) / len(self.y_real)
        print("THE MSE :", mse)
        return mse
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
        Eval=Evaluation(y_predicted=y_pred,y_real=self.y)
        Eval.R_Square()
        Eval._mse(self.m,self.b)     
Linear_regress=Linear_regression(X=[1,2,3,4,5,6,7,8],y=[2,4,6,8,10,12,14,16],Normalized="YES")
Linear_regress.predict(20)
Linear_regress.Eval()

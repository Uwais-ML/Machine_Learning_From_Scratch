class Methods:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.len=len(y)
        self.checkpoint=0
        self._state_counter=0
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
    def Adaptive_lr(self, lr, loss, threshold):
        if loss < threshold:
            adjusted_lr = lr * 0.5
            self._state_counter += 1  
        else:
            adjusted_lr = lr
            
        return adjusted_lr 
    def l1_reg(self,lam,weights):
        return lam*(sum((weight for weight in weights))**2)
    def l2_reg(self,lam,weights):
        return lam*(sum((abs(weight for weight in weights))))
    

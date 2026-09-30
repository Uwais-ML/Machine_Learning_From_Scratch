import math
class Regression_loss:
    def __init__(self, ypred, yactual):
        self.ypred = ypred
        self.yactual = yactual
        self.len = len(self.ypred)
        
    def mse(self):
       
        return sum((self.yactual[i] - self.ypred[i]) ** 2 for i in range(self.len)) / self.len
        
    def mae(self):
       
        return sum(abs(self.yactual[i] - self.ypred[i]) for i in range(self.len)) / self.len
        
    def msle(self):
       
        return sum((math.log(self.yactual[i] + 1) - math.log(self.ypred[i] + 1)) ** 2 for i in range(self.len)) / self.len
        
    def logcosh(self):

        return sum(math.log(math.cosh(self.ypred[i] - self.yactual[i])) for i in range(self.len)) / self.len

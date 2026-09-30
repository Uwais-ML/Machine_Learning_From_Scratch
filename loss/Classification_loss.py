import math
class Classification_loss:
    
    def __init__(self, ypred, yactual):
        self.ypred = ypred
        self.yactual = yactual
        self.len = len(self.ypred)

    def bce(self):
        eps = 1e-15
        total_loss = 0
        for i in range(self.len):
            p = max(eps, min(1 - eps, self.ypred[i]))
            y = self.yactual[i]
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return total_loss / self.len
    
    def focal_loss(self, gamma=2.0):
        eps = 1e-15
        total_loss = 0
        for i in range(self.len):
            sample_loss = 0
            for j in range(len(self.ypred[i])):
                p = max(eps, min(1 - eps, self.ypred[i][j]))
                y = self.yactual[i][j]
                sample_loss += y * ((1 - p) ** gamma) * math.log(p)
            total_loss += -sample_loss
        return total_loss / self.len
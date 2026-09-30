from Architecture_ML.Micrograd import Micrograd
import random
class Neuron:
    def __init__(self,nin):
        self.w=[Micrograd((random.uniform(-1,1)))for _ in range(nin)]
        self.b= Micrograd(random.uniform(-1,1))
    def __call__(self,x):
        weighted_sum=sum((wi*xi for wi,xi in zip(self.w ,x)),self.b)
        out=weighted_sum.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]
class Layer:
    def __init__(self,nin,nout):
        self.Neurons=[Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        outs=[n(x) for n in self.Neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for neuron in self.Neurons for p in neuron.parameters()]
class MLP:
    def __init__(self,nin,nouts):
        self.siz=[nin]+nouts
        self.layers=[Layer(self.siz[i],self.siz[i+1]) for i in range(len(nouts))]
    def __call__(self,x):
        for layers in self.layers:
           x=layers(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0





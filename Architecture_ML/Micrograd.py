import math
class Micrograd:
    def __init__(self,data,children=(),_op=''):
        self.data=data
        self.children=list(children)
        self._backward=lambda:None
        self.grad=0
    def zero_grad(self):
        self.grad=0
    def __add__(self,other):
        other = other if isinstance(other, Micrograd) else Micrograd(other)
        out=Micrograd(self.data+ other.data,(self,other),'+')
        def backward():
            self.grad+=1*out.grad
            other.grad+=1*out.grad

        out._backward=backward
        return out
    def __neg__(self): 
        return self * -1

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): 
        return Micrograd(other) + (-self)
    def __mul__(self,other):
        other = other if isinstance(other, Micrograd) else Micrograd(other)
        out=Micrograd(self.data*other.data,(self,other),'*')
        def backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=backward
        return out
    def __rmul__(self, other):
        return self.__mul__(other)
    def __radd__(self, other):
        return self.__add__(other)
    def __pow__(self, other): 
        if not isinstance(other, (int, float)):
            raise TypeError("Exponents must be int or float for this engine")
        
        out = Micrograd(self.data ** other, (self,), f'**{other}')
        
        def backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
            
        out._backward = backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Micrograd(t, (self,), 'tanh')
        def backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = backward
        
        return out

    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()  


    
class Micrograd:
    def __init__(self, data, children=(), _op=''):
        self.grad = 0.0
        self.op = _op
        self.data = data
        self.children = set(children)
        self._backward = lambda: None

    def __add__(self, other):
        out = Micrograd(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Micrograd(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __tanh__(self):
        x = self.data
        out_data = ((2.71828**x) - (2.71828**-x)) / ((2.71828**x) + (2.71828**-x))
        out = Micrograd(out_data, (self,), 'tanh')
        
        def _backward():
            self.grad += (1.0 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)    
        build_topo(self)
        
        for node in reversed(topo):
            node._backward()

a = Micrograd(2.0)
b = Micrograd(3.0)
c = Micrograd(-8.0)
d = a + b
e = a * b
f = e + c

f.backward()
print("Gradients [f, e, c, b, a]:")
print(f.grad, e.grad, c.grad, b.grad, a.grad)
from Deeplearning.Neural_Network import MLP
X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

y = [1.0, -1.0, -1.0, 1.0] 


n = MLP(3, [4, 4, 1])

for k in range(1000):
    ypred = [n(x) for x in X]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(y, ypred))
    n.zero_grad()
    loss.backward()
    
    learning_rate = 0.05
    for p in n.parameters():
        p.data -= learning_rate * p.grad
        
    print(f"Step {k} | Loss: {loss.data:.4f}")

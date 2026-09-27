# ML Fundamentals

A comprehensive repository for learning and implementing core machine learning concepts from the ground up. Built entirely from scratch to understand the mechanics behind algorithms.

## 📋 Overview

This repository provides clear implementations of fundamental machine learning algorithms using NumPy. Perfect for students, researchers, and practitioners looking to build a strong foundation in ML and understand how things work under the hood without relying on heavy high-level ML frameworks.

## ✨ Features

- **NumPy-based Implementations**: Algorithms built from scratch using NumPy for efficient numerical computations
- **Supervised Learning**: Linear Regression, Logistic Regression, and classification algorithms
- **Deep Learning**: Neural networks and backpropagation implementations
- **Automatic Differentiation**: Micrograd - a tiny autodiff engine for computing gradients
- **RAG Pipeline**: Full retrieval-augmented generation with HTTP and raw IO support
- **Model Checkpointing**: Save and load model states
- **Evaluation Metrics**: Accuracy, confusion matrix, MSE, R² and more
- **Production Ready**: HTTP APIs and deployment-ready components

## 📚 Topics Covered

### Supervised Learning
- **Linear Regression** - Gradient descent optimization with normalization support
- **Logistic Regression** - Binary classification with sigmoid activation
- Evaluation metrics (accuracy, confusion matrix, MSE, R²)

### Deep Learning & Neural Networks
- Neural Network architectures
- Backpropagation and gradient computation
- Activation functions (tanh, sigmoid, ReLU)
- Model training and optimization

### Automatic Differentiation
- **Micrograd** - Tiny autodiff engine for computing gradients automatically
- Forward and backward passes
- Computational graph construction

### RAG Pipeline
- Retrieval-Augmented Generation
- HTTP API endpoints
- Raw IO and data processing
- Integration with neural networks

### Model Evaluation & Validation
- Accuracy metrics
- Confusion Matrix
- Mean Squared Error (MSE)
- R² Score
- Early Stopping

### Optimization
- Gradient Descent
- Learning rate scheduling
- Model checkpointing

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
NumPy
pip
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-fundamentals.git
cd ml-fundamentals
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Examples

Start with the Jupyter notebooks in the `notebooks/` directory:
```bash
jupyter notebook
```

Or run individual Python scripts:
```bash
python scripts/linear_regression.py
```

## 📁 Repository Structure

```
ml-fundamentals/
├── supervised/                  # Supervised learning algorithms
│   ├── Linear_regression.py     # Linear regression from scratch
│   ├── logistics_regression.py  # Logistic regression classifier
│   └── evaluation.py            # Evaluation metrics and utilities
│
├── deep-learning/              # Neural networks and deep learning
│   ├── neural_networks.py       # Neural network implementations
│   ├── Micrograd.py             # Automatic differentiation engine
│   └── layers.py                # Network layers and activations
│
├── rag-pipeline/               # Retrieval-Augmented Generation
│   ├── retrieval.py             # Retrieval components
│   ├── generation.py            # Generation models
│   ├── api.py                   # HTTP API endpoints
│   └── io_handler.py            # Raw IO and data processing
│
├── checkpoints/                # Model checkpoints
│   └── Linear_model_*           # Saved model states
│
├── examples/                   # Example usage and tutorials
└── requirements.txt            # Python dependencies
```

## 💻 Usage Examples

### Linear Regression

```python
from supervised.Linear_regression import Linear_regression

# Create and train model
model = Linear_regression(
    X=[1, 2, 3, 4, 5, 6, 7, 8],
    y=[2, 4, 6, 8, 10, 12, 14, 16],
    Normalized="YES"
)

# Train with early stopping
model.fit(lr=0.005, epoches=100000)

# Make predictions
prediction = model.predict(20)
print(f"Prediction: {prediction}")

# Evaluate model
model.Eval()
```

### Logistic Regression

```python
from supervised.logistics_regression import Logistics_regression, Evaluate

# Training data
X = [30, 32, 35, 37, 38, 40, 41, 42, ...]  # Features
y = [0, 0, 0, 0, 0, 0, 0, 0, ...]          # Binary labels

# Create and train
model = Logistics_regression(X, y)
model.fit(epoch=100000, lr=0.01)

# Make predictions
prediction = model.predict(90)
print(f"Prediction: {prediction}")
```

### Automatic Differentiation with Micrograd

```python
from deep_learning.Micrograd import Micrograd

# Create computational graph
a = Micrograd(2.0)
b = Micrograd(3.0)
c = Micrograd(-8.0)

# Operations
d = a + b
e = a * b
f = e + c

# Backpropagation
f.backward()

# Access gradients
print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")
print(f"Gradient of c: {c.grad}")
```

## 📊 Datasets

The repository includes:
- **Synthetic Data**: Custom-generated datasets for testing algorithms
- **Real-world Examples**: Age/test score correlations, binary classification problems
- **Integration Tests**: Datasets for validating RAG pipeline components
- **Custom Datasets**: Easily add your own data in `/datasets/`

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📖 Learning Path

Recommended order for mastering the material:

1. **Linear Regression** (`supervised/Linear_regression.py`)
   - Understand gradient descent
   - Learn normalization and scaling
   - Explore MSE and R² metrics

2. **Logistic Regression** (`supervised/logistics_regression.py`)
   - Binary classification fundamentals
   - Sigmoid activation function
   - Confusion matrix and accuracy metrics

3. **Automatic Differentiation** (`deep_learning/Micrograd.py`)
   - How gradients flow through computation graphs
   - Building blocks for neural networks
   - Chain rule and backpropagation

4. **Neural Networks** (`deep_learning/neural_networks.py`)
   - Multi-layer perceptrons
   - Activation functions
   - Training and optimization

5. **RAG Pipeline** (`rag_pipeline/`)
   - Retrieval mechanisms
   - Integration with neural networks
   - Production deployment with HTTP APIs

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🎯 Key Implementations

### Gradient Descent Variants
- Full-batch gradient descent with learning rate scheduling
- Early stopping to prevent overfitting
- Model checkpointing for resuming training

### Advanced Features
- **Normalization**: Min-max scaling for stable training
- **Automatic Differentiation**: Compute gradients through any computational graph
- **Model Persistence**: Save and load model checkpoints
- **HTTP API**: Deploy models as web services
- **RAG Integration**: Combine retrieval with generative models

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources & References

- Andrew Ng's Machine Learning Course
- "Hands-On Machine Learning" by Aurélien Géron
- Karpathy's "Neural Networks: Zero to Hero"
- Scikit-learn Documentation
- StatQuest with Josh Starmer (YouTube)

## ❓ FAQ

**Q: Why implement everything from scratch?**
A: Understanding the mechanics is crucial. These implementations help you see exactly how algorithms work, making you a better ML engineer.

**Q: Can I use these in production?**
A: The core algorithms are production-ready. The RAG pipeline includes HTTP APIs for deployment. However, for large-scale production, consider integrating with optimized libraries.

**Q: What's Micrograd?**
A: It's a tiny automatic differentiation engine that computes gradients automatically. It's the foundation for understanding how neural networks learn.

**Q: Do I need advanced math?**
A: Basic calculus and linear algebra help, but the code is well-commented and educational.

**Q: What about the RAG pipeline?**
A: It's a full retrieval-augmented generation system with HTTP endpoints and raw IO support for production deployment.

**Q: What level is this suitable for?**
A: Beginners to intermediate practitioners. Assumes basic Python knowledge.

## 🗂️ File Guide

### Supervised Learning
- **`Linear_regression.py`**: Complete linear regression implementation with normalization, gradient descent, and early stopping
- **`logistics_regression.py`**: Binary logistic regression with sigmoid activation and evaluation metrics

### Deep Learning
- **`Micrograd.py`**: Tiny autodiff engine supporting basic operations (+, *) and tanh activation
- **Neural Networks** (coming soon): Full neural network layers and architectures

### RAG Pipeline
- **Retrieval Components**: Search and ranking algorithms
- **Generation**: Integration with neural networks
- **HTTP API**: REST endpoints for production deployment
- **Raw IO**: Low-level data handling for efficiency

## 🚀 Getting Started Quickly

```bash
# Clone the repo
git clone <your-repo-url>
cd ml-fundamentals

# Run linear regression example
python supervised/Linear_regression.py

# Run logistic regression example
python supervised/logistics_regression.py

# Test automatic differentiation
python deep_learning/Micrograd.py
```

## 🔮 What's Coming

- Enhanced neural network implementations
- Convolutional layers for image processing
- Recurrent architectures (LSTM, GRU)
- Advanced RAG pipeline features
- Model deployment guides

## 📧 Contact & Support

- **Issues**: Open an issue on GitHub for bugs or questions
- **Pull Requests**: Contributions welcome!
- **Email**: your-email@example.com
- **Discussions**: Use GitHub Discussions for general questions

---

Happy Learning! 🎓 Feel free to star ⭐ the repository if you find it helpful.

*Built with ❤️ for the ML community*
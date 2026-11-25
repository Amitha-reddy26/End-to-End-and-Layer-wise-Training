import torch
from src.train import CNN

def test_mnist_model():
    model = CNN(1)
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 10)

def test_cifar_model():
    model = CNN(3)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)

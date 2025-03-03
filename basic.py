import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

writer = SummaryWriter("runs/labs3")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 5
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer, writer, epoch):
    size = len(dataloader.dataset)
    train_loss, correct = 0.0, 0
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    
    train_loss /= len(dataloader)
    accuracy = correct / size
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    return train_loss, accuracy
    


def test_loop(dataloader, model, loss_fn, writer, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    return test_loss, accuracy

epochs = 50
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n---------------")
    train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer, writer, epoch)
    test_loss, test_acc = test_loop(test_dataloader, model, loss_fn, writer, epoch)
    print(f"Epoch {epoch+1} summary: Train Loss: {train_loss:.4f} | Train Accuracy: {100*train_acc:.1f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {100*test_acc:.1f}%")
    writer.add_scalars('TRAIN/Metrics', {'Loss': train_loss, 'Accuracy': train_acc}, epoch)
    writer.add_scalars('TEST/Metrics', {'Loss': test_loss, 'Accuracy': test_acc}, epoch)
print("Done!")
writer.close()

torch.save(model.state_dict(), "model_weights.pth")
print("Saved model weights!")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0]
print(type(x), type(y))
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

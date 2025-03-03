import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

learning_rate = 1e-2
batch_size = 128
epochs = 50
writer = SummaryWriter("runs/mnist")

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: torch.flatten(x))
])

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms,
    target_transform=v2.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

val_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=transforms,
    target_transform=v2.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=transforms,
    target_transform=v2.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

train_dataloader = DataLoader(train_data, batch_size)
val_dataloader = DataLoader(val_data, batch_size)
test_dataloader = DataLoader(val_data, batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_modules = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        logits = self.seq_modules(x)
        return logits


model = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()
softmax = nn.Softmax(dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0.0, 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = softmax(logits)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        predicted = pred.argmax(dim=1)
        acutal = y.argmax(dim=1)
        correct += (predicted == acutal).type(torch.float).sum().item()
    
    train_loss /= num_batches
    accuracy = correct / size

    return train_loss, accuracy


def eval_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    eval_loss, correct = 0.0, 0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            pred = softmax(logits)
            loss = loss_fn(pred, y)

            eval_loss += loss.item()

            predicted = pred.argmax(dim=1)
            acutal = y.argmax(dim=1)
            correct += (predicted == acutal).type(torch.float).sum().item()
        
    eval_loss /= num_batches
    accuracy = correct / size

    return eval_loss, accuracy


for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')

    train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = eval_loop(val_dataloader, model, loss_fn)
    
    writer.add_scalars('TRAIN/Metrics', {'Loss': train_loss, 'Accuracy': train_acc}, epoch+1)
    writer.add_scalars('VAL/Metrics', {'Loss': val_loss, 'Accuracy': val_acc}, epoch+1)
    print(f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} | val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

writer.close()
print("Done!")

# eval mode

test_acc = 0

model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = softmax(logits)
        loss = loss_fn(pred, y)

        predicted = pred.argmax(dim=1)
        acutal = y.argmax(dim=1)
        test_acc += (predicted == acutal).type(torch.float).sum().item()

test_acc /= len(test_dataloader.dataset)
print(f"정확률={test_acc*100}")

classes = [
    "0 - zero",
    "1 - one",
    "2 - two",
    "3 - three",
    "4 - four",
    "5 - five",
    "6 - six",
    "7 - seven",
    "8 - eight",
    "9 - nine",
]

model.eval()
x, y = test_data[0]
with torch.no_grad():
    x, y = x.to(device), y.to(device)
    logits = model(x)
    pred = nn.Softmax()(logits)
    predicted, actual = classes[pred.argmax(dim=0)], classes[y.argmax(dim=0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
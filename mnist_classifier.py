import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. Import libraries (already done above)

# 2. Load MNIST using torchvision.datasets
def load_mnist_data():
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# 3. Build neural networks
class SimpleCNN(nn.Module):
    """Small CNN for MNIST classification"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class TwoLayerFC(nn.Module):
    """2-layer fully connected network"""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(TwoLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 4. Training loop
def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# 5. Evaluate accuracy
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            predictions.extend(pred.cpu().numpy().flatten())
            true_labels.extend(target.cpu().numpy().flatten())
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy, predictions, true_labels

# 6. Plot predictions
def plot_predictions(model, test_loader, num_images=10):
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        predictions = output.argmax(dim=1)
        
        for i in range(num_images):
            # Convert tensor to numpy for plotting
            img = data[i].cpu().squeeze().numpy()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {target[i].item()}, Pred: {predictions[i].item()}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    
    # Choose model (CNN or FC)
    print("\nChoose model:")
    print("1. Small CNN")
    print("2. 2-Layer Fully Connected")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        model = SimpleCNN().to(device)
        print("Using Small CNN")
    else:
        model = TwoLayerFC().to(device)
        print("Using 2-Layer Fully Connected")
    
    # Print model architecture
    print(f"\nModel architecture:\n{model}")
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 5
    
    # Storage for plotting
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc, _, _ = test_model(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print("-" * 60)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)
    
    # Plot some predictions
    print("Plotting predictions...")
    plot_predictions(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved as 'mnist_model.pth'")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import ast  # To convert string representation of lists into actual lists

# Load and preprocess dataset
data = pd.read_csv('dataset.csv')

# Convert 'image' column to lists of integers and stack them into tensor
data['image'] = data['image'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))

# Extract labels and features
X = torch.stack(data['image'].tolist())  # Stack list of tensors into a single tensor
y = torch.tensor((data['state'] == 'open').astype(int).values, dtype=torch.long)  # Encode 'open' as 1, 'closed' as 0

# Define the custom dataset
class BlinkDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx].reshape(26, 34).unsqueeze(0)  # Reshape to (1, 26, 34)
        if self.transform:
            image = self.transform(image)
        label = self.y[idx]
        return image, label

# Apply necessary transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to fit model requirements
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Create the dataset and split it
dataset = BlinkDataset(X, y, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a pretrained model and modify for binary classification
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for 1 channel input
model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming two classes for blink detection

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')
        model.train()

# Train the model
train_model(model, train_loader, val_loader, epochs=10)
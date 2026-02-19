import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split 
import os

# --- ARCHITECTURE (On garde la même) ---
class TrafficLightNet(nn.Module):
    def __init__(self):
        super(TrafficLightNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 4) 
        )
    def forward(self, x): return self.classifier(self.features(x))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")
    
    # Augmentation de données (rend le modèle plus robuste)
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5), # Miroir
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Changement lumière
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    # Pour la validation, on ne fait pas d'augmentation, juste resize
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # Chargement global
    data_dir = 'dataset_cls/train' # Contient Bosch + Ton Background
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    
    # --- SPLIT AUTOMATIQUE 80% / 20% ---
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # On applique le transform "clean" à la validation (astuce technique)
    val_data.dataset.transform = val_transform 
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    
    print(f"Classes : {full_dataset.classes}")
    print(f"Entraînement sur {train_size} images | Validation sur {val_size} images")
    
    model = TrafficLightNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(15): # Un peu plus d'epochs car plus de données
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation sur le subset Bosch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f} | Val Accuracy (Bosch) = {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "traffic_classifier.pth")

    print(f"✅ Mapping final : {full_dataset.class_to_idx}")

if __name__ == "__main__":
    train()
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os

def validate_model(model, loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy

def train_model(model, train_loader, val_loader, epochs=5, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_epoch = -1

    model_path = '/u/scratch/s/sujit009/metastatic_tissue_classification/hub'

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            # While training the inception mdodel, throw away auxiliary output
            outputs, _ = model(images)
            # outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * running_correct / running_total
        val_acc = validate_model(model, val_loader, device=device)

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"- Average Training Loss: {avg_train_loss:.4f} "
              f"- Training Accuracy: {train_acc:.2f}% "
              f"- Validation Accuracy: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc
            }, f'{model_path}/best_model_epoch_{best_epoch}.pth')

    df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })

    df.to_csv('metrics.csv', index=False)
    print(f"Metrics saved to metrics.csv. Best validation accuracy ({best_val_acc:.2f}%) at epoch {best_epoch}")

    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

    plt.figure()
    plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('training_validation_accuracy.png')
    plt.show()

    return model

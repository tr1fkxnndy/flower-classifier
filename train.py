
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# -----------------------------
# Dataset loading
# -----------------------------
def load_dataset(data_dir):

    norm = (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    train_ds = datasets.ImageFolder(
        train_path,
        transform=train_tf
    )

    val_ds = datasets.ImageFolder(
        valid_path,
        transform=val_tf
    )

    if train_ds.classes != val_ds.classes:
        raise ValueError("Training and validation classes do not match!")

    classes = train_ds.classes

    print("Training images:", len(train_ds))
    print("Validation images:", len(val_ds))
    print("Flower categories:", len(classes), flush=True)

    return train_ds, val_ds, classes


# -----------------------------
# Create optimized MobileNetV2
# -----------------------------
def create_model(num_classes):

    print("Loading MobileNetV2...", flush=True)

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2

    model = models.mobilenet_v2(weights=weights)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    print("Model initialized!", flush=True)

    return model


# -----------------------------
# Training
# -----------------------------
def train_model(model, train_loader, val_loader, epochs, device, classes):

    criterion = nn.CrossEntropyLoss()

    # Train only classifier parameters
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=0.001
    )

    best_accuracy = 0.0

    os.makedirs("model", exist_ok=True)

    for epoch in range(epochs):

        print(
            f"\nStarting Epoch {epoch + 1}/{epochs}",
            flush=True
        )

        # Training
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_index, (images, labels) in enumerate(train_loader):

            print(
                f"Training batch {batch_index + 1}/{len(train_loader)}",
                flush=True
            )

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total

        # Validation
        print("Starting validation...", flush=True)

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total

        print("\nEpoch Results")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "img_size": 224
            }, "model/model.pth")

            print("Best model saved!", flush=True)

    print("\nTraining complete!", flush=True)
    print(f"Best validation accuracy: {best_accuracy:.2f}%")


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    args = parser.parse_args()

    # CPU optimization
    torch.set_num_threads(2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device, flush=True)

    # Load dataset
    train_ds, val_ds, classes = load_dataset(args.data_dir)

    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = create_model(len(classes))
    model = model.to(device)

    print("Starting training...", flush=True)

    # Train
    train_model(
        model,
        train_loader,
        val_loader,
        args.epochs,
        device,
        classes
    )


if __name__ == "__main__":
    main()

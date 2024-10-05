import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# Assuming the images are already saved in a directory
def load_existing_images(image_dir):
    """
    Load the paths of pre-generated images from the given directory.
    """
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))
    return image_files


def git(martian_data):
    print("Preparing DataLoader for Martian data (self-training)...")
    martian_data_loader = prepare_data_for_training(
        martian_data,
        labels=[0] * len(martian_data),  # Placeholder labels
        time_labels=[0] * len(martian_data),  # Placeholder time labels
    )
    return martian_data_loader


def prepare_data_for_training(image_files, labels, time_labels, batch_size=32):
    """Prepare data for model training given image files"""
    if not image_files:
        return None  # Early exit if no images are provided
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    image_tensors = [
        transform(Image.open(img))
        for img in image_files
        if os.path.exists(img) and img.endswith(".png")
    ]
    if image_tensors:
        X_tensor = torch.stack(image_tensors)
        y_event_tensor = torch.tensor(labels, dtype=torch.long)
        y_time_tensor = torch.tensor(time_labels, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_event_tensor, y_time_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return None


def prepare_lunar_data_for_training(
    lunar_data, lunar_labels_encoded, lunar_arrival_times_numeric
):
    print("Splitting lunar data into training and validation sets...")
    X_train, X_val, y_event_train, y_event_val, y_time_train, y_time_val = (
        train_test_split(
            lunar_data,
            lunar_labels_encoded,
            lunar_arrival_times_numeric,
            test_size=0.2,
            random_state=42,
        )
    )

    print("Preparing DataLoader for lunar training data...")
    train_loader = prepare_data_for_training(X_train, y_event_train, y_time_train)
    val_loader = prepare_data_for_training(X_val, y_event_val, y_time_val)

    if train_loader is not None:
        print("DataLoader successfully created for training.")
    else:
        print("Error: DataLoader creation failed.")

    return train_loader, val_loader


def prepare_unlabeled_data_loader(image_files, batch_size=32):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    image_tensors = [
        transform(Image.open(img))
        for img in image_files
        if os.path.exists(img) and img.endswith(".png")
    ]
    if image_tensors:
        X_tensor = torch.stack(image_tensors)
        dataset = TensorDataset(X_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return None

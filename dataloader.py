import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

class DataLoaderHandler:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def prepare_data_for_training(self, image_files, time_labels):
        """
        Prepares a DataLoader for training with image files and corresponding time labels.
        """
        if len(image_files) == 0 or len(time_labels) == 0:
            raise ValueError("Image files and time labels cannot be empty")

        # Transformations: convert to grayscale, resize, normalize images
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image_tensors = []
        for img_path in image_files:
            if os.path.exists(img_path) and img_path.endswith('.png'):
                try:
                    img_tensor = transform(Image.open(img_path))
                    image_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            else:
                print(f"Image file not found or invalid: {img_path}")

        if not image_tensors:
            raise ValueError("No valid images were processed")

        # Stack tensors and create PyTorch DataLoader
        X_tensor = torch.stack(image_tensors)
        y_time_tensor = torch.tensor(time_labels, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_time_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def prepare_unlabeled_data_loader(self, image_files):
        """
        Prepares a DataLoader for unlabeled data (image files only).
        """
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image_tensors = []
        for img in image_files:
            if os.path.exists(img) and img.endswith('.png'):
                try:
                    img_tensor = transform(Image.open(img))
                    image_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Error loading image {img}: {e}")
            else:
                print(f"Image file not found or invalid: {img}")

        if image_tensors:
            X_tensor = torch.stack(image_tensors)
            dataset = TensorDataset(X_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        else:
            raise ValueError("No valid images found.")

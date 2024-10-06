import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class SpectrogramDataLoader:
    def __init__(self, image_files, labels, time_labels, batch_size=32):
        self.image_files = image_files
        self.labels = labels
        self.time_labels = time_labels
        self.batch_size = batch_size

    def get_data_loader(self):
        """Prepare data for model training given image files"""
        if not self.image_files:
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
            for img in self.image_files
            if os.path.exists(img) and img.endswith(".png")
        ]
        if image_tensors:
            X_tensor = torch.stack(image_tensors)
            y_time_tensor = torch.tensor(self.time_labels, dtype=torch.float32)
            y_event_tensor = torch.tensor(self.labels, dtype=torch.long)
            dataset = TensorDataset(X_tensor, y_event_tensor, y_time_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return None

    def get_unlabeled_data_loader(self):
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
            for img in self.image_files
            if os.path.exists(img) and img.endswith(".png")
        ]
        if image_tensors:
            X_tensor = torch.stack(image_tensors)
            dataset = TensorDataset(X_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return None

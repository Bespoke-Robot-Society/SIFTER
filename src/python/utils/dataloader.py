import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset


class SpectrogramDataLoader:
    def __init__(self, image_files, time_labels, labels=None, batch_size=32):
        self.image_files = image_files
        self.labels = labels
        self.time_labels = time_labels
        self.batch_size = batch_size

    def get_data_loader(self):
        """Prepare data for model training given image files"""
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        image_tensors = []
        for img in self.image_files:
            # Verify that the file exists and is a .png image
            if os.path.exists(img) and img.endswith(".png"):
                try:
                    img_tensor = transform(Image.open(img))
                    image_tensors.append(img_tensor)
                    print(f"Loaded image: {img}")  # Debug print for loaded images
                except Exception as e:
                    print(f"Error loading image {img}: {e}")
            else:
                print(
                    f"Image file not found or invalid: {img}"
                )  # If the file is not found or not a .png

        # Convert the list of image tensors into a single tensor
        if image_tensors:
            X_tensor = torch.stack(image_tensors)
            y_event_tensor = torch.tensor(self.labels, dtype=torch.long)
            y_time_tensor = torch.tensor(self.time_labels, dtype=torch.float32)

            dataset = TensorDataset(X_tensor, y_event_tensor, y_time_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            print("No valid images found.")
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

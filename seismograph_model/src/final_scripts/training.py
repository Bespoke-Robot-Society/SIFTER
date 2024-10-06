import torch
import pandas as pd
from datetime import timedelta

class ModelTrainer:
    def __init__(self, cnn_model, criterion_time, optimizer):
        self.cnn_model = cnn_model  # Keep model on CPU
        self.criterion_time = criterion_time
        self.optimizer = optimizer

    def train(self, train_loader, num_epochs=10):
        """
        Train the model using lunar data.
        """
        self.cnn_model.train()  # Set the model to training mode

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in train_loader:
                # Unpack batch depending on the structure returned by the dataloader
                if isinstance(batch, (list, tuple)):
                    inputs, time_labels = batch[0], batch[1]
                else:
                    inputs = batch
                    time_labels = None  # Adjust as per the actual data structure

                self.optimizer.zero_grad()

                # Forward pass
                time_output = self.cnn_model(inputs)

                # Compute the loss using criterion_time
                if time_labels is not None:
                    loss = self.criterion_time(time_output, time_labels)

                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate the loss
                    running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
            
    def self_train_on_martian_data(self, martian_data_loader, criterion_time, num_epochs=10):
        """
        Fine-tune the lunar model on Martian data using time prediction with pseudo-labeling.
        """
        self.cnn_model.train()  # Set the model to training mode

        for epoch in range(num_epochs):
            running_loss_time = 0.0
            for i, batch in enumerate(martian_data_loader):
                # Unpack batch depending on the structure returned by the dataloader
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.cnn_model(inputs)

                # Use the model's output as the pseudo-labels for self-supervised learning
                pseudo_labels = outputs.detach()

                # Compute the time prediction loss using the pseudo-labels
                loss_time = criterion_time(outputs, pseudo_labels)

                # Backward pass and optimization
                loss_time.backward()
                self.optimizer.step()

                # Accumulate losses for monitoring
                running_loss_time += loss_time.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss Time: {running_loss_time / len(martian_data_loader)}")


    def evaluate(self, test_loader, scaler, reference_time):
        """
        Evaluate the model on test data, converting relative times to absolute times.
        """
        self.cnn_model.eval()  # Set to evaluation mode
        total_loss = 0
        all_times_abs = []  # Store absolute arrival times

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch depending on the structure returned by the dataloader
                if isinstance(batch, (list, tuple)):
                    inputs, time_labels = batch[0], batch[1]
                else:
                    inputs = batch
                    time_labels = None  # Adjust as per the actual data structure

                # Forward pass
                time_output = self.cnn_model(inputs)

                # Un-normalize the predicted relative times
                unnormalized_time_output = scaler.inverse_transform(time_output.cpu().numpy().reshape(-1, 1))

                # Convert relative to absolute times
                for predicted_time_rel in unnormalized_time_output:
                    predicted_time_abs = (reference_time + timedelta(seconds=float(predicted_time_rel))).strftime('%Y-%m-%dT%H:%M:%S.%f')
                    all_times_abs.append(predicted_time_abs)

                # Compute the loss
                if time_labels is not None:
                    loss = self.criterion_time(time_output, time_labels)
                    total_loss += loss.item()

        # Save the absolute arrival times to CSV
        self.save_predictions_to_csv(all_times_abs)

        print(f"Test Loss: {total_loss / len(test_loader)}")

    def save_predictions_to_csv(self, predictions_abs):
        """
        Save the predicted absolute arrival times to a CSV file.
        """
        df = pd.DataFrame(predictions_abs, columns=['Predicted Arrival Time (absolute)'])
        df.to_csv('predictions_abs_times.csv', index=False)
        print(f"Saved predicted absolute arrival times to CSV.")

    def save_cnn_model(self, path):
        """
        Save the entire model to the specified path.
        """
        torch.save(self.cnn_model, path)
        print(f"Saved the full model to {path}.")

    def save_cnn_model_state_dict(self, path):
        """
        Save only the model's state_dict to the specified path.
        """
        torch.save(self.cnn_model.state_dict(), path)
        print(f"Saved the model's state_dict to {path}.")

    def load_model_state_dict(self, path):
        """
        Load the model's state_dict from the specified path.
        """
        self.cnn_model.load_state_dict(torch.load(path))
        print(f"Loaded the model's state_dict from {path}.")

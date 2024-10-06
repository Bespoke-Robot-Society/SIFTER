import torch

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

    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        """
        self.cnn_model.eval()  # Set to evaluation mode
        total_loss = 0
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

                # Compute the loss
                if time_labels is not None:
                    loss = self.criterion_time(time_output, time_labels)
                    total_loss += loss.item()

        print(f"Test Loss: {total_loss / len(test_loader)}")


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

    def save_cnn_model(self, path):
        """
        Save the entire model to the specified path.
        """
        torch.save(self.cnn_model, path)

    def save_cnn_model_state_dict(self, path):
        """
        Save only the model's state_dict to the specified path.
        """
        torch.save(self.cnn_model.state_dict(), path)

    def load_model_state_dict(self, path):
        """
        Load the model's state_dict from the specified path.
        """
        self.cnn_model.load_state_dict(torch.load(path))

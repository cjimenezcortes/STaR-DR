#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler


# In[2]:


from autoencoder import Autoencoder
from evaluation import Evaluation
from mlp import MLP
from utils import MODEL_FOLDER


# In[3]:


class DeepDRA(nn.Module):
    """
    DeepDRA (Deep Drug Response Anticipation) is a neural network model composed of two autoencoders for cell and drug modalities
    and an MLP for integrating the encoded features and making predictions.

    Parameters:
    - cell_modality_sizes (list): Sizes of the cell modality features.
    - drug_modality_sizes (list): Sizes of the drug modality features.
    - cell_ae_latent_dim (int): Latent dimension for the cell autoencoder.
    - drug_ae_latent_dim (int): Latent dimension for the drug autoencoder.
    - mlp_input_dim (int): Input dimension for the MLP.
    - mlp_output_dim (int): Output dimension for the MLP.
    """

    def __init__(self, cell_modality_sizes, drug_modality_sizes, cell_ae_latent_dim, drug_ae_latent_dim):
        super(DeepDRA, self).__init__()

        # Initialize cell and drug autoencoders
        self.cell_autoencoder = Autoencoder(sum(cell_modality_sizes), cell_ae_latent_dim)
        self.drug_autoencoder = Autoencoder(sum(drug_modality_sizes), drug_ae_latent_dim)

        # Store modality sizes
        self.cell_modality_sizes = cell_modality_sizes
        self.drug_modality_sizes = drug_modality_sizes

        # Compute the MLP input dimension: concatenation of the latent spaces
        mlp_input_dim = cell_ae_latent_dim + drug_ae_latent_dim

        # Initialize MLP
        self.mlp = MLP(cell_ae_latent_dim+drug_ae_latent_dim, 1)

    def forward(self, cell_x, drug_x, return_latent=True):
        """
        Parameters:
        - cell_x (torch.Tensor): Input tensor for cell modality.
        - drug_x (torch.Tensor): Input tensor for drug modality.

        Returns:
        - cell_decoded (torch.Tensor): Decoded tensor for the cell modality.
        - drug_decoded (torch.Tensor): Decoded tensor for the drug modality.
        - mlp_output (torch.Tensor): Output tensor from the MLP.
        """
        # Encode and decode cell modality
        cell_encoded = self.cell_autoencoder.encoder(cell_x)
        cell_decoded = self.cell_autoencoder.decoder(cell_encoded)

        # Encode and decode drug modality
        drug_encoded = self.drug_autoencoder.encoder(drug_x)
        drug_decoded = self.drug_autoencoder.decoder(drug_encoded)

        # Concatenate encoded cell and drug features and pass through MLP
        mlp_output = self.mlp(torch.cat((cell_encoded, drug_encoded), 1))

        #return cell_decoded, drug_decoded, mlp_output
        if return_latent:
            return cell_decoded, drug_decoded, mlp_output, (cell_encoded, drug_encoded)
        else:
            return cell_decoded, drug_decoded, mlp_output

    def compute_l1_loss(self, w):
        """
        Computes L1 regularization loss.

        Parameters:
        - w (torch.Tensor): Input tensor.

        Returns:
        - loss (torch.Tensor): L1 regularization loss.
        """
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        """
        Computes L2 regularization loss.

        Parameters:
        - w (torch.Tensor): Input tensor.

        Returns:
        - loss (torch.Tensor): L2 regularization loss.
        """
        return torch.square(w).sum()


def train(model, train_loader, val_loader, num_epochs, class_weights):
    """
    Trains the DeepDRA (Deep Drug Response Anticipation) model.

    Parameters:
    - model (DeepDRA): The DeepDRA model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - num_epochs (int): Number of training epochs.
    """


    autoencoder_loss_fn = nn.MSELoss()
    mlp_loss_fn = torch.nn.BCELoss()

    train_accuracies = []
    val_accuracies = []

    train_loss = []
    val_loss = []

    mlp_optimizer = optim.Adam(model.parameters(), lr=0.0005,)
    scheduler = lr_scheduler.ReduceLROnPlateau(mlp_optimizer, mode='min', factor=0.8, patience=5, verbose=True)

    # Define weight parameters for each loss term
    cell_ae_weight = 1.0
    drug_ae_weight = 1.0
    mlp_weight = 1.0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total_samples = 0
        for batch_idx, (cell_data, drug_data, target) in enumerate(train_loader):
            mlp_optimizer.zero_grad()

            # Forward pass
            cell_decoded_output, drug_decoded_output, mlp_output = model(cell_data, drug_data, return_latent=False)

            # Compute class weights for the current batch
            # batch_class_weights = class_weights[target.long()]
            # mlp_loss_fn = nn.BCEWithLogitsLoss(weight=batch_class_weights)

            # Compute losses
            cell_ae_loss = cell_ae_weight * autoencoder_loss_fn(cell_decoded_output, cell_data)
            drug_ae_loss = drug_ae_weight * autoencoder_loss_fn(drug_decoded_output, drug_data)
            mlp_loss = mlp_weight * mlp_loss_fn(mlp_output, target)

            # Total loss is the sum of autoencoder losses and MLP loss
            total_loss = drug_ae_loss + cell_ae_loss + mlp_loss

            # Backward pass and optimization
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            mlp_optimizer.step()
            total_train_loss += total_loss.item()         

            # Calculate accuracy
            train_predictions = torch.round(mlp_output)
            train_correct += (train_predictions == target).sum().item()
            train_total_samples += target.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for val_batch_idx, (cell_data_val, drug_data_val, val_target) in enumerate(val_loader):
                cell_decoded_output_val, drug_decoded_output_val, mlp_output_val = model(cell_data_val, drug_data_val, return_latent=False)
                # batch_class_weights = class_weights[val_target.long()]
                # mlp_loss_fn = nn.BCEWithLogitsLoss(weight=batch_class_weights)

                # Compute losses
                cell_ae_loss_val = cell_ae_weight * autoencoder_loss_fn(cell_decoded_output_val, cell_data_val)
                drug_ae_loss_val = drug_ae_weight * autoencoder_loss_fn(drug_decoded_output_val, drug_data_val)
                mlp_loss_val = mlp_weight * mlp_loss_fn(mlp_output_val, val_target)

                # Total loss is the sum of autoencoder losses and MLP loss
                total_val_loss += (drug_ae_loss_val + cell_ae_loss_val + mlp_loss_val).item()
                
                # Calculate accuracy
                val_predictions = torch.round(mlp_output_val)
                correct += (val_predictions == val_target).sum().item()
                total_samples += val_target.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)



        train_accuracy = train_correct / train_total_samples
        train_accuracies.append(train_accuracy)
        val_accuracy = correct / total_samples
        val_accuracies.append(val_accuracy)

        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}'.format(
                epoch + 1, num_epochs, avg_train_loss, avg_val_loss, train_accuracy,
                val_accuracy))

        # Learning rate scheduler step
        scheduler.step(avg_val_loss)

    # Save the trained model
    torch.save(model.state_dict(), MODEL_FOLDER + 'DeepDRA.pth')

    Evaluation.plot_train_val_loss(train_loss, val_loss, num_epochs)


def test(model, test_loader, show_plot=False, return_raw=False):
    """
    Evaluate the model on the test_loader and return metrics computed by Evaluation.evaluate.
    - return_latent is forced to False to obtain only three outputs
    (cell_decoded, drug_decoded, mlp_output).
    - Predictions and labels are aggregated across all batches.
    """
    model.eval()
    device = next(model.parameters()).device

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for test_cell_batch, test_drug_batch, labels_batch in test_loader:
            # Move to device
            test_cell_batch = test_cell_batch.to(device)
            test_drug_batch = test_drug_batch.to(device)
            labels_batch    = labels_batch.to(device)
            # Forward
            _, _, mlp_output = model(test_cell_batch, test_drug_batch, return_latent=False)
            
            scores = mlp_output
            
            all_scores.append(scores)
            all_labels.append(labels_batch)

    # Concat across all batches
    y_score = torch.cat(all_scores, dim=0)
    y_true = torch.cat(all_labels, dim=0)

    results = Evaluation.evaluate(y_true, y_score, show_plot=show_plot)
    
    if return_raw:
        return results, y_true, y_score
    return results


# In[ ]:





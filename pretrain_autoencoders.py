# =============================
# pretrain_autoencoders.py (Phase 1 of STaR-DR)
# =============================
"""
Pretraining of cell and drug autoencoders (AE) for drug-response prediction.

This script trains two independent autoencoders:
- a cell autoencoder
- a drug autoencoder

on a source dataset (CTRP+GDSC), with optional external testing on a different dataset (CCLE).

Main steps:
1) Load raw omics + drug feature matrices (per modality) using RawDataLoader.
2) Build unique entity matrices:
   - X_cell: one row per unique cell line
   - X_drug: one row per unique drug
3) Normalize inputs using feature-wise L2 norms computed on the training split
4) Train autoencoders with a reconstruction objective (MSE).
   - If is_test=False: k-fold cross-validation training on the training dataset.
   - If is_test=True : train on source, evaluate reconstruction on external test data.
5) Save encoder weights to disk.
6) Provide simple qualitative / quantitative diagnostics:
   - t-SNE visualization of raw and latent spaces
   - kNN radius statistics in latent space (compactness proxy)

Outputs:
- Saved model weights (encoder_cell.pth, encoder_drug.pth)
- Loss curves, t-SNE plots.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from data_loader import RawDataLoader
from utils import DATA_MODALITIES, RAW_BOTH_DATA_FOLDER, BOTH_SCREENING_DATA_FOLDER, CCLE_RAW_DATA_FOLDER, CCLE_SCREENING_DATA_FOLDER, TCGA_DATA_FOLDER
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import random
import numpy as np

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
RANDOM_SEED=42

class SimpleAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder.

    Parameters:
    - input_dim (int): Number of input features.
    - latent_dim (int): Dimension of the latent representation.

    Methods:
    - forward(x, return_latent=False):
        Return reconstruction, and optionally the latent vector z.
    """
    def __init__(self, input_dim, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        # Encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        # Decoder architecture
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, return_latent=False):
        """
        Forward pass through the autoencoder.

        Parameters:
        - x (torch.Tensor): Input batch of shape (batch_size, input_dim).
        - return_latent (bool): If True, also return latent vectors z.

        Return:
        - decoded (torch.Tensor): Reconstruction of shape (batch_size, input_dim).
        - z (torch.Tensor, optional): Latent representation of shape (batch_size, latent_dim).
        """
        z = self.encoder(x)
        decoded = self.decoder(z)
        if return_latent:
            return decoded, z
        else:
            return decoded

def train(model, train_loader, val_loader, epochs, alpha=1):
    """
    Train an autoencoder using a (weighted) MSE reconstruction loss.

    Parameters:
    - model (nn.Module): Autoencoder model.
    - train_loader (DataLoader): Loader for training batches.
    - val_loader (DataLoader): Loader for validation batches.
    - epochs (int): Number of epochs.
    - alpha (float): Weight applied to non-zero targets in the weighted MSE.
                    alpha=1 corresponds to standard MSE.

    Return:
    - train_loss (list[float]): Mean training loss per epoch.
    - val_loss (list[float]): Mean validation loss per epoch.
    """
    ...

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0.0

        for batch, data in enumerate(train_loader):
            data = data[0]
            recon = model(data)
            loss = weighted_mse_loss(recon, data, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss.append(total_train_loss / len(train_loader))

        # Validation
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_batch, val_data in enumerate(val_loader):
                val_data = val_data[0]
                val_recon = model(val_data)

                val_batch_loss = weighted_mse_loss(val_recon, val_data, alpha)
                total_val_loss += val_batch_loss.item()

        val_loss.append(total_val_loss / len(val_loader))

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

    return train_loss, val_loss

def weighted_mse_loss(recon, target, alpha=1):
    """
    Motivation:
    Many omics and drug feature matrices can be sparse. Standard MSE tends to
    be dominated by abundant zeros. This loss optionally increases the weight
    of errors on non-zero entries.
    
    Parameters:
    - recon (torch.Tensor): Reconstruction output (batch_size, n_features).
    - target (torch.Tensor): Ground-truth input (batch_size, n_features).
    - alpha (float): alpha > 1 increases importance of non-zero entries.
                     alpha = 1 gives standard MSE.

    Return: loss (torch.Tensor): Scalar loss (mean over batch and features).
    
    """
    alpha = torch.tensor(alpha, dtype=target.dtype, device=target.device)
    one = torch.tensor(1.0, dtype=target.dtype, device=target.device)
    weight = torch.where(target != 0, alpha, one)
    loss = ((recon - target) ** 2 * weight).mean()
    return loss


def plot_tsne(X_tensor, title):
    """
    Plot a 2D t-SNE projection of high-dimensional vectors.

    This is a qualitative diagnostic used to visualize the structure of:
    - raw input space (cell/drug features),
    - latent space produced by the autoencoders.

    Parameters:
    - X_tensor (array-like): Data matrix of shape (n_samples, n_features).
    - title (str): Figure title.

    Return: None (displays a matplotlib figure).
    """
    ...
    
    np.random.seed(42)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_tensor)
    plt.figure(figsize=(7, 5))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, alpha=0.6)
    plt.title(title)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def knn_radius_stats(Z, k: int = 10):
    """
    Compute local radius statistics in latent space:
    - For each point, compute the distance to its k-th nearest neighbor.
    - Returns mean (float), std (float), and coefficient of variation (cv = std/mean) (float).

    Parameters:
    - Z (np.ndarray or torch.Tensor): Latent matrix of shape (n_samples, latent_dim).
    - k (int): Neighbor rank used to define the radius (k-th NN).
    
    Return:
    - mean_r (float): Mean kNN radius.
    - std_r (float): Standard deviation of kNN radii.
    - cv_r (float): Coefficient of variation = std / mean.
    """
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()
    Zs = StandardScaler().fit_transform(Z)

    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean")  # k+1 because the 0-th neighbor is the point itself
    nbrs.fit(Zs)
    dists, _ = nbrs.kneighbors(Zs)
    radii = dists[:, k]  # distance to the k-th nearest neighbor

    mean_r = float(np.mean(radii))
    std_r  = float(np.std(radii))
    cv_r   = float(std_r / (mean_r + 1e-12))
    return mean_r, std_r, cv_r

def train_autoencoder(X_train, X_test, X_train_sizes, X_test_sizes, latent_dim, save_path, epochs=25, batch_size=64):
    """
    Train a SimpleAutoencoder on a training set and evaluate reconstruction on a test set.

    This function:
    1) Instantiates a SimpleAutoencoder with the requested latent_dim.
    2) Converts pandas DataFrames to torch tensors.
    3) Normalizes inputs using feature-wise L2 norms computed on X_train
       (and applies the same scaling to X_test) with a safety threshold.
    4) Trains using the 'train()'.
    5) Saves the model weights to 'save_path'.
    6) Plots training/validation loss curves.

    Parameters:
    - X_train (pd.DataFrame): Training matrix (n_train, n_features).
    - X_test (pd.DataFrame): Test matrix (n_test, n_features).
    - X_train_sizes, X_test_sizes: Unused (kept for API consistency / future use).
    - latent_dim (int): Latent dimension for the AE.
    - save_path (str): Path where the model state_dict is saved.
    - epochs (int): Number of epochs.
    - batch_size (int): Batch size.

    Return: model (SimpleAutoencoder): Trained model (on device).
    """
    model = SimpleAutoencoder(X_train.shape[1], latent_dim).to(device)

    # Convert training data and testing data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train.values)
    X_test_tensor = torch.Tensor(X_test.values)
    
    # normalize data
    thr = 1e-6
    train_norms = torch.norm(X_train_tensor, dim=0, keepdim=True)
    safe_norms  = torch.where(train_norms < thr, torch.ones_like(train_norms), train_norms)
    X_train_tensor = X_train_tensor / safe_norms
    X_test_tensor  = X_test_tensor  / safe_norms

    # Create a TensorDataset with the input features
    train_dataset = TensorDataset(X_train_tensor.to(device))
    test_dataset = TensorDataset(X_test_tensor.to(device))
    
    # Create the train_loader and val_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train_loss, test_loss = train(model, train_loader, val_loader, epochs)

    torch.save(model.state_dict(), save_path)

    # Plot of the loss curve
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs+1), test_loss, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation AE Losses - {save_path}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return model

def cv_train_autoencoder(X_train, X_train_sizes, latent_dim, save_path, device, epochs=25, k=5, batch_size=64):
    """
    Train a SimpleAutoencoder using K-fold cross-validation (CV) on X_train.

    For each fold:
    1) Create train/val samplers.
    2) Normalize data using feature-wise L2 norms computed on the fold's train split.
    3) Train the AE with 'train()' and monitor fold validation loss.
    4) Save model weights to 'save_path'.

    Note:
    - the model is re-initialized each fold and the saved checkpoint corresponds to the last fold.

    Parameters:
    - X_train (pd.DataFrame): Training matrix (n_samples, n_features).
    - X_train_sizes: Unused (kept for API consistency / future use).
    - latent_dim (int): Latent dimension for the AE.
    - save_path (str): Output path for the state_dict.
    - device (torch.device): Device for training.
    - epochs (int): Number of epochs per fold.
    - k (int): Number of folds.
    - batch_size (int): Batch size.

    Returns:
    - model (SimpleAutoencoder): Model instance from the last fold.
    """

    splits = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)


    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(X_train)))):
        print('Fold {}'.format(fold + 1))

        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        model = SimpleAutoencoder(X_train.shape[1], latent_dim)

        model = model.to(device)

        # Convert training data to PyTorch tensors
        X_train_tensor = torch.Tensor(X_train.values)
        
        # normalize data
        thr = 1e-6
        train_norms = torch.norm(X_train_tensor[train_idx], dim=0, keepdim=True)
        safe_norms  = torch.where(train_norms < thr, torch.ones_like(train_norms), train_norms)
        X_train_tensor = X_train_tensor / safe_norms
        
        X_train_tensor = X_train_tensor.to(device)

        # Create a TensorDataset with the input features and target labels
        train_dataset = TensorDataset(X_train_tensor)

        # Create the train_loader and val_loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=len(X_train), sampler=val_sampler)
        
        # Train the model
        train_loss, val_loss = train(model, train_loader, val_loader, epochs, alpha=1)

        torch.save(model.state_dict(), save_path)
        

        # Plot of the loss curve
        plt.figure(figsize=(8,5))
        plt.plot(range(1, epochs+1), train_loss, label='Train Loss')
        plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation AE Losses - {save_path}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return model

def run(k, is_test=False ):
    """
    Orchestrate AE pretraining

    Steps:
    1) Load source training data (CTRP+GDSC) using RawDataLoader.load_data.
    2) Optionally load external test data (CCLE) if is_test=True and intersect features.
    3) Extract unique entity matrices (cells and drugs) using RawDataLoader.get_unique_entities:
       - X_cell_train: unique cell representations
       - X_drug_train: unique drug representations
    4) Print dataset shapes.
    5) Plot t-SNE of raw feature spaces.
    6) For each run i in range(k):
       - Train cell AE
       - Train drug AE
       - Encode the training entities to obtain latent vectors z_cell and z_drug
       - Compute kNN radius statistics
       - Plot t-SNE in latent space

    Parameters:
    - k (int): Number of repeated training runs.
    - is_test (bool): If True, train on source and evaluate reconstruction on external test data.

    Return: None (training, plotting, saving weights).
    """

    # Load raw data
    train_data, _ = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                            raw_file_directory=RAW_BOTH_DATA_FOLDER,
                                            screen_file_directory=None,
                                            sep="\t")

    print(train_data.keys())
    for key, df in train_data.items():
        print(f"{key}: {df.shape}")    

    # Load test data if applicable
    if is_test:
        test_data, _ = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                raw_file_directory=CCLE_RAW_DATA_FOLDER,
                                                screen_file_directory=None,
                                                sep="\t")

        print(test_data.keys())
        for key, df in test_data.items():
            print(f"{key}: {df.shape}")
        
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)


    # Prepare input data for training
    X_cell_train, X_drug_train, cell_train_sizes, drug_train_sizes = RawDataLoader.get_unique_entities(train_data)

    if is_test:
        X_cell_test, X_drug_test, cell_test_sizes, drug_test_sizes = RawDataLoader.get_unique_entities(test_data)

    print('x_cell_train shape:', X_cell_train.shape)
    print('x_drug_train shape:', X_drug_train.shape)

    if is_test:
        print('x_cell_test shape:', X_cell_test.shape)
        print('x_drug_test shape:', X_drug_test.shape)

    # plot t-SNE raw data
    plot_tsne(X_cell_train.values, "t-SNE on raw train data - cells")
    plot_tsne(X_drug_train.values, "t-SNE on raw train data - drugs")
    if is_test:
        plot_tsne(X_cell_test.values, "t-SNE on raw test data - cells")
        plot_tsne(X_drug_test.values, "t-SNE on raw test data - drugs")
    
    # Loop over k runs
    for i in range(k):
        print('Run {}'.format(i))

        if is_test:

            # Train and evaluate the reconstruction on test data
            ae_cell = train_autoencoder(X_cell_train, X_cell_test, cell_train_sizes, cell_test_sizes, 
                                        latent_dim=700, save_path="encoder_cell.pth", epochs=25, batch_size=64)
            ae_drug = train_autoencoder(X_drug_train, X_drug_test, drug_train_sizes, drug_test_sizes, 
                                        latent_dim=50, save_path="encoder_drug.pth", epochs=25, batch_size=64)

        else:
            # Train and evaluate the reconstruction on a train_test_split of the training dataset
            ae_cell = cv_train_autoencoder(X_train=X_cell_train, X_train_sizes=cell_train_sizes, 
                                           latent_dim=700, save_path="encoder_cell.pth", device=device, epochs=25, k=5, batch_size=64)
            ae_drug = cv_train_autoencoder(X_train=X_drug_train, X_train_sizes=drug_train_sizes, 
                                           latent_dim=50, save_path="encoder_drug.pth", device=device, epochs=25, k=5, batch_size=64)

        with torch.no_grad():
            
            x_cell_tensor = torch.tensor(X_cell_train.values, dtype=torch.float32).to(device)
            x_drug_tensor = torch.tensor(X_drug_train.values, dtype=torch.float32).to(device)
            
            _, z_cell = ae_cell(x_cell_tensor, return_latent=True)
            _, z_drug = ae_drug(x_drug_tensor, return_latent=True)

        print("z_cell shape:", z_cell.shape)
        print("z_drug shape:", z_drug.shape)

        # Visualisation t-SNE latent space
        plot_tsne(z_cell.detach().cpu().numpy(), "t-SNE - encoded cells (train data)")
        plot_tsne(z_drug.detach().cpu().numpy(), "t-SNE - encoded drugs (train data)")
        
        # knn radius
        c_mean, c_std, c_cv = knn_radius_stats(z_cell, k=10)
        d_mean, d_std, d_cv = knn_radius_stats(z_drug, k=10)
        print(f"[cells] kNN radius: mean={c_mean:.4f}, std={c_std:.4f}, cv={c_cv:.4f}")
        print(f"[drugs] kNN radius: mean={d_mean:.4f}, std={d_std:.4f}, cv={d_cv:.4f}")


if __name__ == "__main__":

    """
    Script entrypoint.
    It launches the Phase-1 autoencoder pretraining pipeline via 'run()'.

    Notes:
    - 'k' controls how many repeated full pretraining runs are performed.
    - If 'is_test=True', the script trains on the source dataset (CTRP+GDSC)
      and evaluates reconstruction on an external dataset (CCLE) after
      intersecting features across datasets.
    - If 'is_test=False', autoencoders are trained with K-fold CV on the source
      dataset only.

    Outputs:
    - encoder weights saved to disk (encoder_cell.pth, encoder_drug.pth)
    - loss curves, t-SNE plots, kNN radius statistics
    """
    
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(10, is_test=False)
    
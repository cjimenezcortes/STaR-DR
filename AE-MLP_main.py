# =============================
# AE-MLP (DeepDRA derived) — Main script
# =============================
"""
This script trains and evaluates a DeepDRA-derived model (AE-MLP) for drug response prediction (DRP).

High-level pipeline
1) Load raw multi-omics (cell) + drug features and the drug screening matrix (labels).
2) Build paired samples (cell-drug) with binary labels (resistant/sensitive).
3) Training loop:
   - Create an internal train/val split or CV folds.
   - Train AE-MLP end-to-end (cell AE + drug AE + MLP head).
4) Optional latent-space visualizations (t-SNE) and local-geometry metric (kNN radius).
5) Evaluate on held-out validation fold(s) or external test set.

Notes:
- Normalization uses per-feature L2 norms computed on train data only, then reused for val/test.
- RUS is applied only on the training portion to preserve natural label distribution for validation/testing.
"""

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from DeepDRA import DeepDRA, train, test
from data_loader import RawDataLoader
from evaluation import Evaluation
from utils import RANDOM_SEED, DATA_MODALITIES, RAW_BOTH_DATA_FOLDER, BOTH_SCREENING_DATA_FOLDER, CCLE_RAW_DATA_FOLDER, CCLE_SCREENING_DATA_FOLDER
import random
import torch
import numpy as np
import pandas as pd
import pickle

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------------
# Global hyperparameters
# -----------------------------

# Mini-batch size used by PyTorch DataLoaders
batch_size = 64

# Number of epochs for model training
num_epochs = 25

# Latent dimensions for each autoencoder branch
cell_latent_dim = 700
drug_latent_dim = 50

# -----------------------------
# Visualization / geometry helpers
# -----------------------------

def tsne_unlabeled(X_np, title):
    """Run a 2D t-SNE on an unlabeled set of vectors and plot a single scatter.
    
    Parameters:
    - X_np (np.ndarray): Array of shape (n_samples, n_features).
    - title (str).
    """
    np.random.seed(42)
    emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_np)
    
    plt.figure(figsize=(7,5))
    plt.scatter(emb[:,0], emb[:,1], s=10, alpha=0.6)
    plt.title(title)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def tsne_labeled(z_tensor, y_tensor, title):
    """Run a 2D t-SNE on latent vectors and color points by binary label.
    
        Parameters:
    - z_tensor (torch.Tensor): Latent matrix of shape (n_samples, d).
    - y_tensor (torch.Tensor): Labels of shape (n_samples, 1) or (n_samples,).
    - title (str).
    """
    z_embedded = TSNE(n_components=2, random_state=42).fit_transform(z_tensor.numpy())
    y_np = y_tensor.detach().cpu().numpy().ravel().astype(int)
    plt.figure(figsize=(8, 6))
    for label, color in zip([0, 1], ['blue', 'red']):
        plt.scatter(z_embedded[y_np == label, 0],
                    z_embedded[y_np == label, 1],
                    label='Resistant' if label == 0 else 'Sensitive',
                    c=color, 
                    s=10, 
                    alpha=0.7
                   )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def knn_radius_stats(Z: np.ndarray, k: int = 10):
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
        
    # Standardize latent features: mean=0, std=1 per dimension
    Zs = StandardScaler().fit_transform(Z)

    # k+1 neighbors because self is neighbor 0
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nbrs.fit(Zs)
    dists, _ = nbrs.kneighbors(Zs)
    radii = dists[:, k]  # distance to the k-th neighbor
    mean_r = float(np.mean(radii))
    std_r  = float(np.std(radii))
    cv_r   = float(std_r / (mean_r + 1e-12))
    return mean_r, std_r, cv_r

@torch.no_grad()
def visualize_unique_latents_from_labeled(model, 
                                          x_cell_train_df, 
                                          x_drug_train_df, 
                                          cell_norms, 
                                          drug_norms, 
                                          device
):
    """
    t-SNE visualization of unique entities (cells/drugs) from labeled training pairs.
    
    - Extract unique cell feature vectors and unique drug feature vectors seen in the labeled train set.
    - Apply the same train-derived L2 norms.
    - Encode cells and drugs using the model encoders.
    - Visualize encoded vectors with t-SNE.
    - Print kNN radius stats on these unique encoded vectors.
    """

    # Unique entities from labeled training pairs
    cell_unique_df = x_cell_train_df.drop_duplicates()
    drug_unique_df = x_drug_train_df.drop_duplicates()

    # Convert to tensors
    x_cell_u = torch.tensor(cell_unique_df.values, dtype=torch.float32)
    x_drug_u = torch.tensor(drug_unique_df.values, dtype=torch.float32)

    # Apply the same norms learned on training data
    x_cell_u = (x_cell_u / cell_norms).to(device)
    x_drug_u = (x_drug_u / drug_norms).to(device)

    model.eval()
    B = 512 # batch size for encoding

    # Encode unique cells
    zc_list = []
    for i in range(0, x_cell_u.size(0), B):
        xb_cell = x_cell_u[i:i+B]
        dummy_drug = torch.zeros((xb_cell.size(0), x_drug_u.shape[1]), device=device)
        _, _, _, (zc_b, _) = model(xb_cell, dummy_drug, return_latent=True)
        zc_list.append(zc_b.detach().cpu())
    z_cell_unique = torch.cat(zc_list, dim=0).numpy()

    # Encode unique drugs
    zd_list = []
    for j in range(0, x_drug_u.size(0), B):
        xb_drug = x_drug_u[j:j+B]
        dummy_cell = torch.zeros((xb_drug.size(0), x_cell_u.shape[1]), device=device)
        _, _, _, (_, zd_b) = model(dummy_cell, xb_drug, return_latent=True)
        zd_list.append(zd_b.detach().cpu())
    z_drug_unique = torch.cat(zd_list, dim=0).numpy()

    # Visualize latent spaces (unlabeled)
    tsne_unlabeled(z_cell_unique, "t-SNE - encoded cells (train data)")
    tsne_unlabeled(z_drug_unique, "t-SNE - encoded drugs (train data)")

    # shapes
    print("z_cell shape:", z_cell_unique.shape)
    print("z_drug shape:", z_drug_unique.shape)
    
    # Unsupervised local geometry metric
    c_mean, c_std, c_cv = knn_radius_stats(z_cell_unique, k=10)
    d_mean, d_std, d_cv = knn_radius_stats(z_drug_unique, k=10)
    print(f"[kNN radius | Cells] mean={c_mean:.4f}, std={c_std:.4f}, cv={c_cv:.4f}")
    print(f"[kNN radius | Drugs] mean={d_mean:.4f}, std={d_std:.4f}, cv={d_cv:.4f}")

# -----------------------------
# Helpers for Leave-Out protocols
# -----------------------------

def parse_cell_ids_from_pairs(index_like):
    """
    Parse cell IDs from an index of pair identifiers formatted as strings.
    - This is used to build group labels for LCO (Leave-Cell-Out) splits.

    Parameters:
    index_like (array-like): Index or list-like object containing pair identifiers (X_pairs.index).

    Returns np.ndarray: Array of extracted cell IDs (dtype=object).
    """
    s = pd.Index(index_like).to_series()
    # Capture everything before the first comma: "(CELL,DRUG)" -> "CELL"
    return s.str.extract(r'^\(([^,]+),')[0].values

def parse_drug_ids_from_pairs(index_like):
    """
    Parse drug IDs from an index of pair identifiers formatted as strings.
    - This is used to build group labels for LDO (Leave-Drug-Out) splits.

    Parameters:
    index_like (array-like): Index or list-like object containing pair identifiers (X_pairs.index).

    Returns np.ndarray: Array of extracted drug IDs (dtype=object).
    """
    s = pd.Index(index_like).to_series()
    # Capture everything after the comma: "(CELL,DRUG)" -> "DRUG"
    return s.str.extract(r'^\([^,]+,\s*([^)]+)\)')[0].values

# -----------------------------
# Training / evaluation functions
# -----------------------------

def train_AEMLP(
    x_cell_train, 
    x_cell_test, 
    x_drug_train, 
    x_drug_test, 
    y_train, y_test, 
    cell_sizes, 
    drug_sizes, 
    device, 
    visualize='first', 
    run_id=0, 
    return_raw=False,
):
    """
    Train and evaluate the AE-MLP model.

    - Internal train/val split is stratified.
    - L2 norms are computed only on the raw train split (before RUS).
    - RUS is applied onlyon the training split.
    - Validation is kept un-resampled.
    - Test set is normalized using training norms.

    Parameters:
    - X_cell_train (pd.DataFrame): Training data for the cell modality.
    - X_cell_test (pd.DataFrame): Test data for the cell modality.
    - X_drug_train (pd.DataFrame): Training data for the drug modality.
    - X_drug_test (pd.DataFrame): Test data for the drug modality.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Test labels.
    - cell_sizes (list): Sizes of the cell modality features.
    - drug_sizes (list): Sizes of the drug modality features.
    - device (torch.device): CUDA or CPU.
    - visualize {'never','first','always'}: Control t-SNE latent visualizations.
     - run_id (int).
     - return_raw (bool): If True, also return raw y_true and y_score from the 'test' function.

    Returns:
    - results (dict) or (results, y_true, y_score): Output of Evaluation.evaluate / DeepDRA.test.
    """
    
    model = DeepDRA(cell_sizes, drug_sizes, cell_latent_dim, drug_latent_dim).to(device)

    # Concatenate pairs so that split/resampling happens at the pair level
    X_pairs = pd.concat([x_cell_train, x_drug_train], axis=1)
    n_cell  = x_cell_train.shape[1]
    y_all   = np.asarray(y_train).ravel()

    # Stratified split of indices (train/val inside the training set)
    idx = np.arange(len(y_all))
    train_idx, val_idx = train_test_split(
        idx, 
        test_size=0.1, 
        random_state=RANDOM_SEED, 
        stratify=y_all
    )

    # Compute normalization factors on raw training subset (before RUS) 
    thr = 1e-6
    X_train = X_pairs.iloc[train_idx]
    
    x_cell_train_tensor = torch.tensor(X_train.iloc[:, :n_cell].values, dtype=torch.float32)
    x_drug_train_tensor = torch.tensor(X_train.iloc[:,  n_cell:].values, dtype=torch.float32)
    
    cell_norms = torch.norm(x_cell_train_tensor, dim=0, keepdim=True)
    cell_norms = torch.where(cell_norms < thr, torch.ones_like(cell_norms), cell_norms)
    
    drug_norms = torch.norm(x_drug_train_tensor, dim=0, keepdim=True)
    drug_norms = torch.where(drug_norms < thr, torch.ones_like(drug_norms), drug_norms)

    # save train norms for reproducibility
    torch.save(cell_norms.detach().cpu(), f"train_cell_l2norms_run_{run_id}.pt")
    torch.save(drug_norms.detach().cpu(), f"train_drug_l2norms_run_{run_id}.pt")
    
    # RandomUnderSampler on the training set
    rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
    X_train_bal, y_train_bal = rus.fit_resample(X_pairs.iloc[train_idx], y_all[train_idx])

    # Reconstruct (cell/drug) for train/val
    x_cell_train = X_train_bal.iloc[:, :n_cell]
    x_drug_train = X_train_bal.iloc[:, n_cell:]
    y_train = y_train_bal

    x_cell_val = X_pairs.iloc[val_idx, :n_cell]
    x_drug_val = X_pairs.iloc[val_idx, n_cell:]
    y_val = y_all[val_idx]

    # shapes + class counts
    print(f"x_cell_train shape (after RUS): {x_cell_train.shape}")
    print(f"x_drug_train shape (after RUS): {x_drug_train.shape}")
    print(f"x_cell_val shape:  {x_cell_val.shape}")
    print(f"x_drug_val shape:  {x_drug_val.shape}")
    print(
        f"y_train counts (after RUS): "
        f"0={np.sum(y_train_bal==0)}, 1={np.sum(y_train_bal==1)}"
    )

    # Convert train data to Pytorch tensors + apply train norms (train/val use the same norms)
    x_cell_train_tensor = torch.Tensor(x_cell_train.values) / cell_norms
    x_drug_train_tensor = torch.Tensor(x_drug_train.values) / drug_norms 
    x_cell_val_tensor = torch.Tensor(x_cell_val.values) / cell_norms
    x_drug_val_tensor = torch.Tensor(x_drug_val.values) / drug_norms 

    y_train_tensor = torch.Tensor(y_train).unsqueeze(1)
    y_val_tensor = torch.Tensor(y_val).unsqueeze(1)

    # Send tensors to device
    x_cell_train_tensor = x_cell_train_tensor.to(device)
    x_drug_train_tensor = x_drug_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)

    x_cell_val_tensor = x_cell_val_tensor.to(device)
    x_drug_val_tensor = x_drug_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)
    
    # Create a TensorDataset with the input features and target labels
    train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_cell_val_tensor, x_drug_val_tensor, y_val_tensor)
    
    # Create the train_loader and val_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the AE-MLP model 
    train(model, train_loader, val_loader, num_epochs, class_weights=None)

    # Optional: visualize unique cell/drug latents from labeled training pairs
    if visualize in ('always',) or (visualize == 'first' and run_id == 0):
        visualize_unique_latents_from_labeled(
            model,
            x_cell_train, 
            x_drug_train,
            cell_norms, 
            drug_norms,
            device
        )
        
    model.eval()

    # Convert test data and norms to PyTorch tensors
    x_cell_test_tensor = torch.Tensor(x_cell_test.values).to(device)
    x_drug_test_tensor = torch.Tensor(x_drug_test.values).to(device)
    y_test_tensor = torch.Tensor(y_test).to(device)

    cell_norms = cell_norms.to(device)
    drug_norms = drug_norms.to(device)

    # Normalize test set using training norms
    x_cell_test_tensor = x_cell_test_tensor / cell_norms
    x_drug_test_tensor = x_drug_test_tensor / drug_norms


    # Create a TensorDataset with the input features and target labels for testing
    test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test), shuffle=False)


    # Optional: label-colored t-SNE on training latents
    if visualize == 'always' or (visualize == 'first' and run_id == 0):
        with torch.no_grad():
            model.eval()
            
            # Encode latents in batches (avoid OOM)
            enc_dataloader = DataLoader(
                TensorDataset(x_cell_train_tensor, x_drug_train_tensor), 
                batch_size=512, 
                shuffle=False
            )
            
            zc_list, zd_list = [], []
            for xb_c, xb_d in enc_dataloader:
                _, _, _, (zc_b, zd_b) = model(xb_c, xb_d, return_latent=True)
                zc_list.append(zc_b.detach().cpu())
                zd_list.append(zd_b.detach().cpu())
                
            z_cell_fold = torch.cat(zc_list, dim=0)
            z_drug_fold = torch.cat(zd_list, dim=0)

        # Visualize latent spaces (labeled)
        tsne_labeled(z_cell_fold, y_train_tensor, "t-SNE - z_cell")
        tsne_labeled(z_drug_fold, y_train_tensor, "t-SNE - z_drug")

        # Visualize concatenated latent spaces
        tsne_labeled(
            torch.cat([z_cell_fold, z_drug_fold], dim=1), 
            y_train_tensor, 
            "t-SNE - z_cell + z_drug"
        )

    # Extract submodules 
    enc_cell = model.cell_autoencoder.encoder     # cell encoder
    enc_drug = model.drug_autoencoder.encoder     # drug encoder
    mlp_head = model.mlp                          # MLP head
    
    # Save the encoders (modules) + the MLP (state_dict)
    run_id = 0  # to adapt if multiple runs
    torch.save(enc_cell, f"encoder_cell_trained_run_{run_id}.pt")
    torch.save(enc_drug, f"encoder_drug_trained_run_{run_id}.pt")
    torch.save(mlp_head.state_dict(), f"mlp_trained_run_{run_id}.pth")

    # Final test evaluation
    return test(model, test_loader, show_plot=True, return_raw=return_raw)

def cv_train(
    x_cell_train, 
    x_drug_train, 
    y_train, 
    cell_sizes, 
    drug_sizes, 
    device, k=5, 
    visualize='first', 
    run_id=0,
    split='RANDOM'
):
    """
    Stratified K-fold cross-validation on labeled cell-drug pairs.

    For each fold:
    - Compute L2 norms on raw train fold only (pre-RUS).
    - Apply RUS on train fold only.
    - Train AE-MLP end-to-end and evaluate on the validation fold.

    Splitting modes: {'RANDOM','LCO','LDO'}
    The 'split' argument controls how folds are formed:

    - "RANDOM" (default):
        Standard StratifiedKFold on pairs (preserves class ratio per fold).

    - "LCO" (Leave-Cell-Out):
        Group-aware stratified CV where all pairs from the same cell are kept in the
        same fold (no cell leakage across train/val within a fold).

    - "LDO" (Leave-Drug-Out):
        Group-aware stratified CV where all pairs from the same drug are kept in the
        same fold (no drug leakage across train/val within a fold).


    Returns: history (dict): Lists of metrics aggregated across folds.
    """
    
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Balanced Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}

    # Concatenate to split/resample by pair
    X_pairs = pd.concat([x_cell_train, x_drug_train], axis=1)
    # Labels as 0/1
    y_np = np.asarray(y_train).ravel()

    # Compute groups if needed:
    groups = None
    # Group for LDO/LCO (LDO: by drug ID or LCO: by cell ID)
    if split.upper() == 'LDO':
        groups = parse_drug_ids_from_pairs(X_pairs.index)
    elif split.upper() == 'LCO':
        groups = parse_cell_ids_from_pairs(X_pairs.index)

    # Select splitter
    if split.upper() in ("LCO", "LDO"):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        split_iter = cv.split(X_pairs, y_np, groups=groups)
    else:
        # StratifiedKFold preserves class proportions across folds
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        split_iter = cv.split(np.zeros(len(y_np)), y_np)

    for fold, (train_data, val_data) in enumerate(split_iter):
        print(f"Fold {fold+1} ({split.upper()})")

        # split by pairs
        X_train, y_train = X_pairs.iloc[train_data], y_np[train_data]
        X_val, y_val = X_pairs.iloc[val_data],  y_np[val_data]
        
        n_cell = x_cell_train.shape[1]
        
        # Build raw train tensors to compute norms 
        x_cell_train_tensor = torch.tensor(X_train.iloc[:, :n_cell].values, dtype=torch.float32)
        x_drug_train_tensor = torch.tensor(X_train.iloc[:, n_cell:].values, dtype=torch.float32)
        
        # Compute L2 norms on train fold
        thr = 1e-6
        cell_norms = torch.norm(x_cell_train_tensor, dim=0, keepdim=True)
        cell_norms = torch.where(cell_norms < thr, torch.ones_like(cell_norms), cell_norms)
        
        drug_norms = torch.norm(x_drug_train_tensor, dim=0, keepdim=True)
        drug_norms = torch.where(drug_norms < thr, torch.ones_like(drug_norms), drug_norms)
    
        # save train norms for reproducibility
        torch.save(cell_norms.detach().cpu(), f"train_cell_l2norms_run_{run_id}.pt")
        torch.save(drug_norms.detach().cpu(), f"train_drug_l2norms_run_{run_id}.pt")
        
        # RUS on train fold
        rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
        X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

        # Separate cells and drugs (train is resampled, val is not)
        x_cell_train = X_train_bal.iloc[:, :n_cell]
        x_drug_train = X_train_bal.iloc[:, n_cell:]
        x_cell_val = X_val.iloc[:, :n_cell]
        x_drug_val = X_val.iloc[:, n_cell:]

        # Shapes and class counts
        print(f"[Fold {fold+1}] x_cell_train shape (after RUS): {x_cell_train.shape}")
        print(f"[Fold {fold+1}] x_drug_train shape (after RUS): {x_drug_train.shape}")
        print(f"[Fold {fold+1}] x_cell_val shape:  {x_cell_val.shape}")
        print(f"[Fold {fold+1}] x_drug_val shape:  {x_drug_val.shape}")
        print(f"[Fold {fold+1}] y_train counts (after RUS): "
              f"0={np.sum(y_train_bal==0)}, 1={np.sum(y_train_bal==1)}")

        # Tensors
        x_cell_train_tensor = torch.Tensor(x_cell_train.values) / cell_norms
        x_drug_train_tensor = torch.Tensor(x_drug_train.values) / drug_norms   
        x_cell_val_tensor = torch.Tensor(x_cell_val.values) / cell_norms
        x_drug_val_tensor = torch.Tensor(x_drug_val.values) / drug_norms   
        y_train_tensor = torch.Tensor(y_train_bal).unsqueeze(1)
        y_val_tensor = torch.Tensor(y_val).unsqueeze(1)
        
        # Send to device + normalization using train norms
        x_cell_train_tensor = x_cell_train_tensor.to(device)
        x_drug_train_tensor = x_drug_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)

        # Send to device
        x_cell_val_tensor = x_cell_val_tensor.to(device)
        x_drug_val_tensor = x_drug_val_tensor.to(device)
        y_val_tensor = y_val_tensor.to(device)
        
        # Create a TensorDataset with the input features and target labels
        train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, y_train_tensor)
        val_dataset   = TensorDataset(x_cell_val_tensor,  x_drug_val_tensor, y_val_tensor)
        
        # Create the train_loader and val_loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        # Initialize model each fold
        model = DeepDRA(cell_sizes, drug_sizes, cell_latent_dim, drug_latent_dim).to(device)

        # Display model
        if fold==0:
            print("\nModel architecture:\n")
            print(model)

        # Train model on this fold
        train(model, train_loader, val_loader, num_epochs, class_weights=None)

        # Optional: visualize unique latents
        if visualize in ('always',) or (visualize == 'first' and fold == 0):
            visualize_unique_latents_from_labeled(
                model,
                x_cell_train, 
                x_drug_train, 
                cell_norms, 
                drug_norms,
                device
            )
            
        # Evaluate on the entire validation fold in a single pass
        val_loader_full = DataLoader(val_dataset, batch_size=len(y_val_tensor), shuffle=False)
        results = test(model, val_loader_full, show_plot=False)
        
        # Aggregate metrics
        Evaluation.add_results(history, results)

        # Optional: label-colored t-SNE on training latents
        # T-SNE (never / first / always)
        if visualize == 'always' or (visualize == 'first' and fold == 0):
            with torch.no_grad():
                model.eval()
                # Encode latent
                enc_dataloader = DataLoader(
                    TensorDataset(x_cell_train_tensor, x_drug_train_tensor), 
                    batch_size=512, 
                    shuffle=False
                )
                zc_list, zd_list = [], []
                for xb_c, xb_d in enc_dataloader:
                    _, _, _, (zc_b, zd_b) = model(xb_c, xb_d, return_latent=True)
                    zc_list.append(zc_b.detach().cpu())
                    zd_list.append(zd_b.detach().cpu())
                    
                z_cell_fold = torch.cat(zc_list, dim=0)
                z_drug_fold = torch.cat(zd_list, dim=0)

            # Visualize latent spaces (labeled)
            tsne_labeled(z_cell_fold, y_train_tensor, "t-SNE - z_cell")
            tsne_labeled(z_drug_fold, y_train_tensor, "t-SNE - z_drug")

            # Visualize concatenated latent spaces
            tsne_labeled(torch.cat([z_cell_fold, z_drug_fold], dim=1), y_train_tensor, "t-SNE - z_cell + z_drug")

    
    # Extract submodules 
    enc_cell = model.cell_autoencoder.encoder     # cell encoder
    enc_drug = model.drug_autoencoder.encoder     # drug encoder
    mlp_head = model.mlp                          # MLP head
    
    # Save the encoders (modules) + the MLP (state_dict)
    run_id = 0  # to adapt
    torch.save(enc_cell, f"encoder_cell_trained_run_{run_id}.pt")
    torch.save(enc_drug, f"encoder_drug_trained_run_{run_id}.pt")
    torch.save(mlp_head.state_dict(), f"mlp_trained_run_{run_id}.pth")
    
    return history

def run(k, is_test=False, split='RANDOM'):
    """
    Main entry point: runs either
    - cross-validation on the training dataset (is_test=False), OR
    - train-on-train + evaluate-on-external-test (is_test=True).

    Parameters:
    - k (int): Number of runs. In CV mode, each run executes a k-fold CV internally.
    - is_test (bool): 
        If True, evaluate on external test dataset (CCLE)
        If False, do cross-validation within the training dataset (CTRP-GDSC)

    Returns:
    - history (dict): Dictionary containing evaluation metrics for each run.
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Initialize a dictionary to store evaluation metrics
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Balanced Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}
    
    # Load training data
    train_data, train_drug_screen = RawDataLoader.load_data(
        data_modalities=DATA_MODALITIES,
        raw_file_directory=RAW_BOTH_DATA_FOLDER,
        screen_file_directory=BOTH_SCREENING_DATA_FOLDER,
        sep="\t"
    )

    print('train_data when loaded:', train_data.keys())
    for key, df in train_data.items():
        print(f"{key}: {df.shape}")

    
    # Load test data if applicable
    if is_test:
        test_data, test_drug_screen = RawDataLoader.load_data(
            data_modalities=DATA_MODALITIES,
            raw_file_directory=CCLE_RAW_DATA_FOLDER,
            screen_file_directory=CCLE_SCREENING_DATA_FOLDER,
            sep="\t"
        )

        print('test_data when loaded:', test_data.keys())
        for key, df in test_data.items():
            print(f"{key}: {df.shape}")

        # Feature intersection to match train/test dimensions
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)


        print('train_data after feature intersection with test set:', train_data.keys())
        for key, df in train_data.items():
            print(f"{key}: {df.shape}")

    # Save feature columns for reproducibility:
    all_features = {}
    for key, df in train_data.items():
        all_features[key] = df.columns.tolist()

    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(all_features, f)
    
    # Prepare input data for training : build paired inputs (X_cell, X_drug) and labels Y from screening matrix
    x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(
        train_data,
        train_drug_screen
    )

    

    if is_test:
        x_cell_test, x_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(
            test_data,
            test_drug_screen
        )
    all_runs = []
    
    # Loop over k runs
    for i in range(k):
        print('Run {}'.format(i))

        if is_test:

            # External test evaluation: train on train, evaluate once on test
            results, y_true, y_score = train_AEMLP(
                x_cell_train, 
                x_cell_test, 
                x_drug_train, 
                x_drug_test, 
                y_train, 
                y_test, 
                cell_sizes, 
                drug_sizes, 
                device, 
                visualize='first', 
                run_id=i, 
                return_raw=True,
            )
            # Display final results
            Evaluation.add_results(history, results)
            all_runs.append((y_true, y_score))

        else:
            # CV evaluation inside train set
            results = cv_train(
                x_cell_train, 
                x_drug_train, 
                y_train, 
                cell_sizes, 
                drug_sizes, 
                device, k=5, 
                visualize='first', 
                run_id=i,
                split=split
            )
            # cv_train returns lists (one per fold) -> extend history
            if isinstance(results.get('AUC', None), list):
                for m in history:
                    history[m].extend(results[m])
            else:
                Evaluation.add_results(history, results)

    # If external test mode: aggregate ROC/PR curves across runs
    if is_test and len(all_runs) > 0:
        fpr_grid, mean_tpr, std_tpr, auc_mean, auc_std = Evaluation.aggregate_roc(all_runs, n_points=200)
        rec_grid, mean_prec, std_prec, auprc_mean, auprc_std = Evaluation.aggregate_pr(all_runs, n_points=200)

        Evaluation.plot_mean_roc(fpr_grid, mean_tpr, std_tpr, auc_mean, auc_std, label="AE-MLP")
        Evaluation.plot_mean_pr(rec_grid, mean_prec, std_prec, auprc_mean, auprc_std, label="AE-MLP")


    
    # Final summary prints
    Evaluation.show_final_results(history)
    return history

# -----------------------------
# Script execution
# -----------------------------
# Choose number of runs, execution mode and split strategy.
#  
#    Number of runs:
#    - The first argument of run(k) controls the number of independent runs.
#    - Each run uses a different random split.
#    - When is_test = False, each run performs a full k-fold cross-validation.
#    - When is_test = True, each run trains once and evaluates on the same external test set.
#
#    Execution mode:
#    - is_test = False   # Cross-validation on the training dataset (internal evaluation)
#    - is_test = True    # Train on training dataset and evaluate once on an external test dataset
#    
#    Split strategy (controls how train/validation splits are built):
#    - split = "RANDOM"  # standard stratified pair split
#    - split = "LCO"     # leave-cell-out 
#    - split = "LDO"     # leave-drug-out 
#
#    NOTE:
#    In the current implementation, Leave-Out protocols (LCO / LDO) are
#    applied ONLY in 'cv_train'. Therefore, when 'is_test = True', the 'split'
#    argument is ignored and the internal train/validation split performed
#    inside 'train_AEMLP' always corresponds to a standard stratified
#    pair-level split ("RANDOM").

if __name__ == '__main__':
    # Ensure reproducibility for python / numpy / torch RNGs
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    run(1, is_test=True, split='RANDOM')
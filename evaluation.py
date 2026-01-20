#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import (
roc_auc_score, 
average_precision_score, 
confusion_matrix, 
f1_score, 
precision_score, 
recall_score, 
accuracy_score, 
roc_curve, 
precision_recall_curve, 
balanced_accuracy_score
)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttest_ind


class Evaluation:
    @staticmethod
    def plot_train_val_accuracy(train_accuracies, val_accuracies, num_epochs):
        """
        Plot training and validation accuracies over epochs.

        Parameters:
        - train_accuracies (list): List of training accuracies.
        - val_accuracies (list): List of validation accuracies.
        - num_epochs (int): Number of training epochs.

        Returns:
        - None
        """
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracies')
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_train_val_loss(train_loss, val_loss, num_epochs):
        """
        Plot training and validation losses over epochs.

        Parameters:
        - train_loss (list): List of training losses.
        - val_loss (list): List of validation losses.
        - num_epochs (int): Number of training epochs.

        Returns:
        - None
        """
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
        plt.legend()
        plt.show()
        
    @staticmethod
    def aggregate_roc(all_runs, n_points=200):
        """
        all_runs: list of (y_true, y_score) over runs
        Returns: fpr_grid, mean_tpr, std_tpr, auc_mean, auc_std
        """
        fpr_grid = np.linspace(0, 1, n_points)
        tpr_mat  = []
    
        aucs = []
        for y_true, y_score in all_runs:
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
            
            # Interpolate TPR on the common FPR grid
            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_interp[0]  = 0.0
            tpr_interp[-1] = 1.0
            
            tpr_mat.append(tpr_interp)
            aucs.append(roc_auc_score(y_true, y_score))
    
        tpr_mat  = np.vstack(tpr_mat)
        mean_tpr = tpr_mat.mean(axis=0)
        std_tpr  = tpr_mat.std(axis=0, ddof=1) if tpr_mat.shape[0] > 1 else np.zeros_like(mean_tpr)
    
        return fpr_grid, mean_tpr, std_tpr, float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs)>1 else 0.0
    
    @staticmethod
    def aggregate_pr(all_runs, n_points=200):
        """
        Aggregate Precision–Recall curves across multiple runs by interpolating
        precision as a function of recall.

        Parameters:
        - all_runs (list of tuples): List of (y_true, y_score) for each run.
        - n_points (int): Number of points used to define the common recall grid.
        
        Returns: recall_grid, mean_prec, std_prec, auprc_mean, auprc_std
        """
        rec_grid = np.linspace(0, 1, n_points)
        prec_mat = []
        auprcs   = []
    
        for y_true, y_score in all_runs:
            prec, rec, _ = precision_recall_curve(y_true, y_score, pos_label=1)

            # Ensure strictly increasing recall for interpolation
            rec_eps, idx = np.unique(rec, return_index=True)
            prec_unique  = prec[idx]
            prec_interp  = np.interp(rec_grid, rec_eps, prec_unique)
            prec_mat.append(prec_interp)
    
            auprcs.append(average_precision_score(y_true, y_score))
    
        prec_mat  = np.vstack(prec_mat)
        mean_prec = prec_mat.mean(axis=0)
        std_prec  = prec_mat.std(axis=0, ddof=1) if prec_mat.shape[0] > 1 else np.zeros_like(mean_prec)
    
        return rec_grid, mean_prec, std_prec, float(np.mean(auprcs)), float(np.std(auprcs, ddof=1)) if len(auprcs)>1 else 0.0
    
    @staticmethod
    def plot_mean_roc(fpr_grid, mean_tpr, std_tpr, auc_mean=None, auc_std=None, label="Model"):
        plt.figure()
        plt.plot(fpr_grid, mean_tpr, lw=2, label=f"{label} (mean AUC={auc_mean:.3f}±{auc_std:.3f})" if auc_mean is not None else label)
        plt.fill_between(fpr_grid, np.maximum(mean_tpr-std_tpr, 0), np.minimum(mean_tpr+std_tpr, 1), alpha=0.2)
        plt.plot([0,1],[0,1],'--', lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC — mean ± std over runs")
        plt.legend(); plt.tight_layout(); plt.show()
    
    @staticmethod
    def plot_mean_pr(rec_grid, mean_prec, std_prec, auprc_mean=None, auprc_std=None, label="Model"):
        plt.figure()
        plt.plot(rec_grid, mean_prec, lw=2, label=f"{label} (mean AUPRC={auprc_mean:.3f}±{auprc_std:.3f})" if auprc_mean is not None else label)
        lower = np.maximum(mean_prec - std_prec, 0)
        upper = np.minimum(mean_prec + std_prec, 1)
        plt.fill_between(rec_grid, lower, upper, alpha=0.2)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall — mean ± std over runs")
        plt.legend(); plt.tight_layout(); plt.show()

    @staticmethod
    def evaluate(all_targets, mlp_output, show_plot=False, threshold=0.5, verbose=True):
        """
        Evaluate model performance based on predictions and targets.

        Parameters:
        - all_targets (numpy.ndarray): True target labels, with 0=resistant, 1=sensitive.
        - mlp_output (numpy.ndarray): Predicted probabilities for class 1 (sensitive)
        - show_plot (bool): Whether to display ROC and PR curves.

        Returns:
        - results (dict): Dictionary containing evaluation metrics.
        """

        # 1) Convert tensors / arrays to 1D numpy arrays
        y_true  = (all_targets.detach().cpu().numpy() if hasattr(all_targets, "detach") else np.asarray(all_targets)).ravel()
        y_score = (mlp_output.detach().cpu().numpy()   if hasattr(mlp_output,   "detach") else np.asarray(mlp_output)).ravel()
        
        # Ensure correct dtypes
        y_true  = y_true.astype(int)     # ensure 0/1
        y_score = y_score.astype(float)  # scores/probabilities
    
        # 2) Binarize predictions using the chosen threshold (by default 0.5)
        y_pred = (y_score >= threshold).astype(int)
    
        # 3) Metrics aligned on positive class (sensitive)
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1  = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
        # Threshold-free metrics (based on probas)
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)  # AP (standard)
    
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        #print("Prevalence(1)=", (y_true == 1).mean(),
        #      f"Frac(score≥{threshold})=", (y_score >= threshold).mean(),
        #      "min/max=", float(y_score.min()), float(y_score.max()))
        if verbose:
            print("Confusion matrix (rows=true, cols=pred) [0,1]:\n", cm)
            print(f'Accuracy: {acc:.3f}, BalancedAcc: {bacc:.3f}, Precision (sensibles=1): {prec:.3f}, '
                  f'Recall (sensibles=1): {rec:.3f}, F1 (sensibles=1): {f1:.3f}, '
                  f'AUC: {auc:.3f}, AUPRC: {auprc:.3f}, Threshold used: {threshold:.3f}')

        if show_plot:
            import matplotlib.pyplot as plt
            # ROC
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (AUC={auc:.3f})')
            plt.plot(fpr, tpr)
            plt.show()
            # PR
            pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_score, pos_label=1)
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title(f'PR Curve (AUPRC={auprc:.3f})')
            plt.plot(pr_rec, pr_prec)
            plt.show()
            # Violin plot (1 = Responder/Sensitive, 0 = Non-responder/Resistant)
            #pred_df = pd.DataFrame({'Prediction': y_score, 'Target': y_true})
            #scores_resp = pred_df.loc[pred_df['Target'] == 1, 'Prediction']
            #scores_non  = pred_df.loc[pred_df['Target'] == 0, 'Prediction']
            #fig, ax = plt.subplots()
            #ax.set_ylabel("score (P[class=1|x])")
            #ax.set_xticks([1, 2])
            #ax.set_xticklabels(['Responder (1: sensitive)', 'Non-responder (0: resistant)'])
            #data_to_plot = [scores_resp, scores_non]
            #plt.ylim(0, 1)
            #from statsmodels.stats.weightstats import ttest_ind
            #p_value = ttest_ind(scores_resp, scores_non)[1]
            #plt.title(f'Score distributions (p≈{p_value:.2e})')
            #bp = ax.violinplot(data_to_plot, showextrema=True, showmeans=True, showmedians=True)
            #bp['cmeans'].set_color('r'); bp['cmedians'].set_color('g')
            #plt.show()

        return {
            'Accuracy': acc,
            'Balanced Accuracy': bacc,
            'Precision': prec,
            'Recall': rec,
            'F1 score': f1,
            'AUC': auc,
            'AUPRC': auprc,
            'Threshold': float(threshold),}

    @staticmethod
    def add_results(result_list, current_result):
        for metric in ['AUC', 'AUPRC', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 score']:
            value = current_result[metric]
            if isinstance(value, (list, np.ndarray)):
                value = float(value[0])
            else:
                value = float(value)

            result_list[metric].append(value)

        return result_list

    @staticmethod
    def show_final_results(result_list):
        print("Final Results:")
        avg_auc = np.mean(result_list['AUC'])
        avg_auprc = np.mean(result_list['AUPRC'])
        std_auprc = np.std(result_list['AUPRC'])
        avg_accuracy = np.mean(result_list['Accuracy'])
        avg_bacc = np.mean(result_list['Balanced Accuracy'])
        avg_precision = np.mean(result_list['Precision'])
        avg_recal = np.mean(result_list['Recall'])
        avg_f1score = np.mean(result_list['F1 score'])
        print(
            f'AVG: Accuracy: {avg_accuracy:.3f}, BalancedAcc: {avg_bacc:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recal:.3f}, F1 score: {avg_f1score:.3f}, AUC: {avg_auc:.3f}, ,AUPRC: {avg_auprc:.3f}')

        print(" Average AUC: {:.3f} \t Average AUPRC: {:.3f} \t Std AUPRC: {:.3f}".format(avg_auc, avg_auprc, std_auprc))
        return {'Accuracy': avg_accuracy, 'Balanced Accuracy': avg_bacc,'Precision': avg_precision, 'Recall': avg_recal, 'F1 score': avg_f1score, 'AUC': avg_auc,
                'AUPRC': avg_auprc}
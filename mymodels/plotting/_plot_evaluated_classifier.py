import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize




def _check_input(y_test, x_test):
    """
    Check if the input y_test is a valid binary or multiclass classification label.
    
    Args:
        y_test: True labels
        
    Returns:
        bool: True if y_test is a valid binary or multiclass classification label, False otherwise
    """
    # Check if y_test is a valid binary or multiclass classification label
    if hasattr(y_test, 'ndim'):
        assert (y_test.ndim == 1), "Input y_test must be a 1D array-like structure."
    else:
        assert isinstance(y_test, (list, np.ndarray, pd.Series)), "Input y_test must be a 1D array-like structure."

    assert len(y_test) == len(x_test), "The length of y_test and x_test must be the same."

    assert not np.any(pd.isnull(y_test)), "Input _y must not contain any empty values."

    return None



def _plot_roc_curve(y_test, x_test, optimal_model_object):
    """Creates ROC curve plot for classification model evaluation.
    
    For binary classification, plots a single ROC curve.
    For multiclass classification, plots one-vs-rest ROC curves for each class.
    
    Returns:
        tuple: (fig, ax) The figure and axis objects containing the ROC curve plot
    """
    _check_input(y_test, x_test)

    # Check if the optimal model object has the predict_proba method
    if not hasattr(optimal_model_object, "predict_proba"):
        raise ValueError("The optimal model object does not have the predict_proba method.")

    # Get probability estimates
    y_prob = optimal_model_object.predict_proba(X=x_test)
    # Get unique classes
    classes = np.unique(y_test)
    n_classes = len(classes)

    fig = plt.figure(figsize=(10, 10))
    ###########################################################################################
    # Binary classification
    if n_classes == 2:
        # For binary classification, use probability of positive class
        if y_prob.shape[1] == 2:  # If there are probabilities for both classes
            # Get probability of positive class (index 1)
            pos_prob = y_prob[:, 1]
        else:
            # If only one probability returned (some models do this for binary)
            pos_prob = y_prob.ravel()
            
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, pos_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal reference line
    ###########################################################################################
    
    ###########################################################################################
    # Multiclass classification
    else:
        # Convert y_test to binary format (one-hot encoding)
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            # For each class, plot one-vs-rest ROC curve
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, 
                        label=f'ROC curve of class {classes[i]} (AUC = {roc_auc[i]:.3f})')
            
        macro_roc_auc = np.mean([roc_auc[i] for i in range(n_classes)])
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.text(0.5, 0.3, f'Macro-average AUC = {macro_roc_auc:.3f}', 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    ###########################################################################################
    
    # Plotting
    # Formatting the plot
    plt.xlabel('False Positive Rate', fontdict={'size': 14})
    plt.ylabel('True Positive Rate', fontdict={'size': 14})
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontdict={'size': 16})
    plt.legend(loc="lower right", prop={'size': 12})
    plt.grid(alpha=0.3)
    
    # Set appearance
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks(size=14)
    plt.yticks(size=14)
    
    # Apply tight layout
    plt.tight_layout()
    
    return fig, ax



def _plot_pr_curve(y_test, x_test, optimal_model_object):
    """Creates PR curve plot for classification model evaluation.
    
    For binary classification, plots a single PR curve.
    For multiclass classification, plots one-vs-rest PR curves for each class.
    
    Returns:
        tuple: (fig, ax) The figure and axis objects containing the PR curve plot
    """
    # Check if the optimal model object has the predict_proba method
    if not hasattr(optimal_model_object, "predict_proba"):
        raise ValueError("The optimal model object does not have the predict_proba method.")

    # Get probability estimates
    y_prob = optimal_model_object.predict_proba(X=x_test)
    # Get unique classes
    classes = np.unique(y_test)
    n_classes = len(classes)

    fig = plt.figure(figsize=(10, 10))
    
    # Binary classification case
    if n_classes == 2:
        # For binary classification, use probability of positive class
        if y_prob.shape[1] == 2:  # If there are probabilities for both classes
            # Get probability of positive class (index 1)
            pos_prob = y_prob[:, 1]
        else:
            # If only one probability returned (some models do this for binary)
            pos_prob = y_prob.ravel()
            
        # Compute Precision-Recall curve and area
        precision, recall, _ = precision_recall_curve(y_test, pos_prob)
        pr_auc = auc(recall, precision)
        
        # Plot PR curve
        plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        
    # Multiclass classification case
    else:
        # Convert y_test to binary format (one-hot encoding)
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # Compute PR curve and PR area for each class
        precision = dict()
        recall = dict()
        pr_auc = dict()
        
        for i in range(n_classes):
            # For each class, plot one-vs-rest PR curve
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
            pr_auc[i] = auc(recall[i], precision[i])
            plt.plot(recall[i], precision[i], lw=2, 
                     label=f'PR curve of class {classes[i]} (AUC = {pr_auc[i]:.3f})')
            
        # Calculate and display macro-average PR AUC
        macro_pr_auc = np.mean([pr_auc[i] for i in range(n_classes)])
        plt.plot([0, 1], [1, 0], 'k--', lw=2)
        plt.text(0.5, 0.3, f'Macro-average AUC = {macro_pr_auc:.3f}', 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    # Plotting
    # Formatting the plot
    plt.xlabel('Recall', fontdict={'size': 14})
    plt.ylabel('Precision', fontdict={'size': 14})
    plt.title('Precision-Recall Curve', fontdict={'size': 16})
    plt.legend(loc="lower left", prop={'size': 12})
    plt.grid(alpha=0.3)
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Set appearance
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks(size=14)
    plt.yticks(size=14)
    
    # Apply tight layout
    plt.tight_layout()
    
    return fig, ax



def _plot_confusion_matrix(y_test, y_test_pred):
    """Creates confusion matrix plot for classification model evaluation.
    
    Args:
        y_test: True labels
        y_test_pred: Predicted labels
        
    Returns:
        tuple: (fig, ax) The figure and axis objects containing the confusion matrix plot
    """
    # Get the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_test, y_test_pred]))
    n_classes = len(classes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
                xticklabels=classes, yticklabels=classes, cbar=True, ax=ax)
    
    # Formatting
    plt.xlabel('Predicted label', fontdict={'size': 14})
    plt.ylabel('True label', fontdict={'size': 14})
    plt.title('Confusion Matrix', fontdict={'size': 16})
    
    # Rotate tick labels if there are too many classes
    if n_classes > 4:
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=45, fontsize=12)
    else:
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    
    # Add text with accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(n_classes/2, -0.1, f'Accuracy: {accuracy:.3f}', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    # Apply tight layout
    plt.tight_layout()
    
    return fig, ax

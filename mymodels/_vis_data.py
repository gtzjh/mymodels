import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import logging
from joblib import Parallel, delayed


def _vis_category(
        data,
        name=None,
        save_dir=None,
        plot_format="jpg",
        plot_dpi=500,
        show=False
    ):
    """Visualize categorical data with a bar chart showing frequency distribution
    
    Args:
        data: A 1D ndarray-like object with categorical data
        name: Optional name for the data (used for plot title and filename)
        save_dir: Directory to save the plot (pathlib.Path object)
        plot_format: Format of the saved plot (default: jpg)
        plot_dpi: DPI of the saved plot (default: 500)
        show: Whether to show the plot (default: False)
    
    Returns:
        str: Path to the saved plot file if save_dir is provided, None otherwise
    """
    # Validate input is 1D ndarray-like
    assert hasattr(data, 'shape') and len(data.shape) == 1, "Data must be a 1D array-like object"
    
    # Convert to numpy array for easier handling
    data_array = np.array(data)
    
    # Instead of asserting, check and warn if there are floating point values
    if np.issubdtype(data_array.dtype, np.floating):
        logging.warning("Categorical data contains floating point numbers. Converting all values to strings for visualization.")
    
    # Handle NaN values specially before converting to strings
    if np.issubdtype(data_array.dtype, np.number):
        # Replace NaN values with "Missing" for better visualization
        data_array = np.array(["Missing" if pd.isna(x) else x for x in data_array])
    
    # Convert all values to strings to avoid comparison between different types
    data_array = np.array([str(x) for x in data_array])
    
    # Calculate unique values and their counts
    unique_values, counts = np.unique(data_array, return_counts=True)
    
    # Get total number of categories
    total_categories = len(unique_values)
    
    # Sort categories by count (descending)
    sorted_indices = np.argsort(-counts)
    sorted_values = unique_values[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Calculate total count for percentage calculations
    total_count = np.sum(counts)
    
    # Limit to 10 categories for visualization
    if total_categories > 10:
        display_values = sorted_values[:10]
        display_counts = sorted_counts[:10]
        
        # Calculate statistics for remaining categories
        remaining_categories = total_categories - 10
        remaining_count = np.sum(sorted_counts[10:])
        remaining_percentage = (remaining_count / total_count) * 100
        
        # Log warning about hidden categories
        logging.warning(f"Showing only top 10 categories out of {total_categories}.")
        logging.warning(f"The remaining {remaining_categories} categories represent {remaining_percentage:.2f}% of the data.")
    else:
        display_values = sorted_values
        display_counts = sorted_counts
        remaining_categories = 0
        remaining_percentage = 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart with dark blue borders and light blue fill with 40% transparency
    bars = ax.bar(
        range(len(display_values)), 
        display_counts, 
        color='#4682B4',  # Light blue
        alpha=0.4,        # 40% transparency
        edgecolor='#00008B',  # Dark blue
        linewidth=1.5     # Border width
    )
    
    # Set x-tick labels to category names (limiting length if too long)
    display_labels = [str(val)[:15] + '...' if len(str(val)) > 15 else str(val) for val in display_values]
    ax.set_xticks(range(len(display_values)))
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    
    # Add value annotations on top of each bar (count and percentage)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (display_counts[i] / total_count) * 100
        
        # Format the annotation with count and percentage
        annotation = f'{height:,} ({percentage:.1f}%)'
        
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + (total_count * 0.01),  # Position slightly above bar
            annotation, 
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set labels and title
    title = "Category Distribution"
    if name:
        title = f"Distribution of {name}"
        ax.set_ylabel("Count")
    else:
        ax.set_ylabel("Count")
    
    ax.set_xlabel("Categories")
    fig.suptitle(title, fontsize=16)
    
    # Add note about hidden categories if any
    if remaining_categories > 0:
        plt.figtext(
            0.5, 0.01, 
            f"Note: {remaining_categories} additional categories not shown ({remaining_percentage:.2f}% of data)", 
            ha='center', 
            fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.5)
        )
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15 if remaining_categories > 0 else 0.1)
    
    # Save the plot if save_dir is provided
    if save_dir:
        # Set filename
        filename = f"category_{name if name else 'data'}.{plot_format}"
        filepath = save_dir / filename
        
        # Ensure the directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(filepath, dpi=plot_dpi)
        
        result = str(filepath)
    else:
        result = None
    
    # Show the plot if required
    if show:
        plt.show()
    
    plt.close()
    
    return result



def _vis_data_distribution(
        data, 
        name=None,
        save_dir=None, 
        plot_format="jpg", 
        plot_dpi=500, 
        show=False
    ):
    """Visualize the data distribution with violin plot and box plot

    Args:
        data: A 1D ndarray-like object to plot
        name: Optional name for the data (used for plot title and filename)
        save_dir: Directory to save the plot (pathlib.Path object)
        plot_format: Format of the saved plot (default: jpg)
        plot_dpi: DPI of the saved plot (default: 500)
        show: Whether to show the plot (default: False)
    
    Returns:
        str: Path to the saved plot file if save_dir is provided, None otherwise
    """
    # Validate input is 1D ndarray-like
    assert hasattr(data, 'shape') and len(data.shape) == 1, "Data must be a 1D array-like object"
    
    # Convert to numpy array for easier handling
    data_array = np.array(data)
    
    # Check if all elements are numeric
    assert np.issubdtype(data_array.dtype, np.number), "All elements in data must be numeric"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create violin plot with steel blue color
    sns.violinplot(y=data_array, ax=ax, inner=None, color='#4682B4')
    
    # Add box plot inside the violin plot
    sns.boxplot(y=data_array, ax=ax, width=0.2, color='white', boxprops=dict(alpha=0.7))
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set labels
    if name:
        ax.set_ylabel(name)
        fig.suptitle(f"Distribution of {name}", fontsize=16)
    else:
        ax.set_ylabel("Value")
        fig.suptitle("Data Distribution", fontsize=16)
    
    ax.set_xlabel("")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if save_dir is provided
    if save_dir:
        # Set filename
        filename = f"distribution_{name if name else 'data'}.{plot_format}"
        filepath = save_dir / filename
        
        # Ensure the directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(filepath, dpi=plot_dpi)
        
        result = str(filepath)
    else:
        result = None
    
    # Show the plot if required
    if show:
        plt.show()
    
    plt.close()
    
    return result



"""A WARNING should occur if some features are highly correlated 
'cause that may influence the model's interpretability."""
def _vis_correlation(
        data,
        corr_threshold=0.8,
        name=None,
        save_dir=None, 
        plot_format="jpg", 
        plot_dpi=500, 
        show=False,
        n_jobs=-1  # Add n_jobs parameter with default value of -1 (use all cores)
    ):
    """Visualize the correlation matrix with heatmaps for both Pearson and Spearman correlations
    
    Args:
        data: A pandas DataFrame containing only numeric columns
        corr_threshold: Threshold for correlation warning (default: 0.8)
        name: Optional name for the plot (used for filename)
        save_dir: Directory to save the plot (pathlib.Path object)
        plot_format: Format of the saved plot (default: jpg)
        plot_dpi: DPI of the saved plot (default: 500)
        show: Whether to show the plot (default: False)
        n_jobs: Number of jobs for parallel computation (default: -1, use all cores)
    
    Returns:
        dict: Paths to the saved plot files if save_dir is provided, None otherwise
    """
    # Validate input
    assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
    
    # Check if all columns are numeric
    numeric_columns = data.select_dtypes(include=np.number).columns
    if len(numeric_columns) < data.shape[1]:
        non_numeric = set(data.columns) - set(numeric_columns)
        logging.warning(f"Dropping non-numeric columns: {non_numeric}")
        data = data[numeric_columns]
    
    # Check if data has at least 2 columns
    if data.shape[1] < 2:
        logging.warning("Data must have at least 2 columns for correlation analysis")
        return None
    
    # Calculate Pearson and Spearman correlations and p-values
    pearson_corr = data.corr(method='pearson')
    spearman_corr = data.corr(method='spearman')
    
    # Calculate p-values
    def calculate_pvalues(df, method='pearson'):
        df_copy = df.copy()
        p_values = pd.DataFrame(np.zeros_like(df_copy.values), index=df_copy.index, columns=df_copy.columns)
        
        # Dictionary to track dropped rows for each column pair
        dropped_rows_info = {}
        
        # Helper function to calculate p-value for a single pair
        def calculate_single_pvalue(i, j):
            col_i_name = df_copy.columns[i]
            col_j_name = df_copy.columns[j]
            
            # Extract the two columns as a new DataFrame
            pair_df = df_copy.iloc[:, [i, j]]
            
            # Get original row count
            original_count = len(pair_df)
            
            # Drop rows where either column has NaN
            pair_df_clean = pair_df.dropna()
            
            # Calculate number of dropped rows
            dropped_count = original_count - len(pair_df_clean)
            
            # Store information about dropped rows
            pair_key = f"{col_i_name}_{col_j_name}"
            dropped_rows_info[pair_key] = {
                'columns': (col_i_name, col_j_name),
                'dropped_count': dropped_count,
                'remaining_count': len(pair_df_clean)
            }
            
            # Calculate correlation and p-value using cleaned data
            if method == 'pearson':
                corr, p_value = scipy.stats.pearsonr(pair_df_clean.iloc[:, 0], pair_df_clean.iloc[:, 1])
            else:  # spearman
                corr, p_value = scipy.stats.spearmanr(pair_df_clean.iloc[:, 0], pair_df_clean.iloc[:, 1])
                
            return (i, j, p_value)
        
        # Generate pairs of column indices
        pairs = [(i, j) for i in range(df_copy.shape[1]) for j in range(i+1, df_copy.shape[1])]
        
        # Process pairs in parallel using joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_single_pvalue)(i, j) for i, j in pairs
        )
        
        # Fill the p_values dataframe with results
        for i, j, p_value in results:
            p_values.iloc[i, j] = p_value
            p_values.iloc[j, i] = p_value  # Make symmetric
            
        return p_values, dropped_rows_info
    
    pearson_pvalues, pearson_dropped_rows = calculate_pvalues(data, method='pearson')
    spearman_pvalues, spearman_dropped_rows = calculate_pvalues(data, method='spearman')
    
    # Check for high correlations and issue warning
    if (abs(pearson_corr) > corr_threshold).any().any():
        high_corr_pairs = []
        for i in range(pearson_corr.shape[0]):
            for j in range(i+1, pearson_corr.shape[1]):
                if abs(pearson_corr.iloc[i, j]) > corr_threshold:
                    high_corr_pairs.append((pearson_corr.index[i], pearson_corr.columns[j], pearson_corr.iloc[i, j]))
        
        # logging.warning(f"High correlations (>{corr_threshold}) detected between features:")
        for col1, col2, corr_val in high_corr_pairs:
            logging.warning(f"  - {col1} and {col2}: {corr_val:.3f}")
    
   
    # Detailed output of the pearson_dropped_rows dictionary
    if pearson_dropped_rows:  # Only output if the dictionary is not empty
        logging.warning("=== Detailed NaN Handling Information for Pearson Correlation ===")
        logging.warning(f"Total column pairs: {len(pearson_dropped_rows)}")
        
        # Sort by dropped count to see pairs with most NaNs first
        sorted_pairs = sorted(pearson_dropped_rows.items(), key=lambda x: x[1]['dropped_count'], reverse=True)
        
        for pair_key, info in sorted_pairs:
            if info['dropped_count'] > 0:  # Only log pairs that actually had rows dropped
                col1, col2 = info['columns']
                drop_percent = 100 * info['dropped_count'] / (info['dropped_count'] + info['remaining_count'])
                logging.warning(f"Pair: {col1} - {col2}, Dropped: {info['dropped_count']} rows ({drop_percent:.2f}%), Remaining: {info['remaining_count']} rows")
    
    # Create heatmaps
    result_paths = {}
    
    # Function to create and save a heatmap
    def create_heatmap(corr_matrix, pvalues, dropped_rows_info, method_name):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Generate mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create the heatmap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            cmap=cmap, 
            vmax=1.0, 
            vmin=-1.0,
            center=0,
            square=True, 
            linewidths=.5, 
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": .5},
            ax=ax
        )
        
        # Annotate with p-values where significant
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                if i > j:  # Lower triangle only
                    if pvalues.iloc[i, j] < 0.05:
                        ax.text(
                            j + 0.5, i + 0.5, 
                            "*", 
                            ha='center', va='center',
                            color='white' if abs(corr_matrix.iloc[i, j]) > 0.4 else 'black',
                            fontweight='bold', fontsize=12
                        )
                    if pvalues.iloc[i, j] < 0.01:
                        ax.text(
                            j + 0.5, i + 0.5, 
                            "**", 
                            ha='center', va='center',
                            color='white' if abs(corr_matrix.iloc[i, j]) > 0.4 else 'black',
                            fontweight='bold', fontsize=12
                        )
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Set title
        plot_title = f"{method_name} Correlation"
        if name:
            plot_title += f" - {name}"
        plt.title(plot_title, fontsize=16)
        
        # Add subtitle about dropped rows
        total_dropped = sum(info['dropped_count'] for info in dropped_rows_info.values())
       
        # Remove top and right spines 
        # (not directly applicable to heatmap but keeping style consistent)
        ax.collections[0].colorbar.outline.set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if save_dir is provided
        if save_dir:
            # Set filename
            filename = f"correlation_{method_name.lower()}"
            if name:
                filename += f"_{name}"
            filename += f".{plot_format}"
            filepath = save_dir / filename
            
            # Ensure the directory exists
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the plot
            plt.savefig(filepath, dpi=plot_dpi)
            
            result = str(filepath)
        else:
            result = None
        
        # Show the plot if required
        if show:
            plt.show()
        
        plt.close()
        
        return result
    
    # Create both heatmaps
    result_paths['pearson'] = create_heatmap(pearson_corr, pearson_pvalues, pearson_dropped_rows, "Pearson")
    result_paths['spearman'] = create_heatmap(spearman_corr, spearman_pvalues, spearman_dropped_rows, "Spearman")
    
    return result_paths

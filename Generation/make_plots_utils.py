import h5py
import pandas as pd
import os

import os
import numpy as np
import pandas as pd

import random

import os
import numpy as np
import pandas as pd
import random


import os
import numpy as np
import pandas as pd
import random


import os
import numpy as np
import pandas as pd
import random
import os
import numpy as np
import pandas as pd
import random
import os
import numpy as np
import pandas as pd
import random

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.image import imread

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

def calculate_subject_metrics(subject_id, base_path):
    """Calculate metrics for a specific subject."""
    subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
    df = load_embeddings(subject_folder)

    target_paths = df[(df['split'] == 'test') & (df['embedding_type'] == 'IMG')]['original_img_path']
    target_embeddings = np.stack(df[(df['split'] == 'test') & (df['embedding_type'] == 'IMG')]['embeddings'])

    eeg_paths = df[(df['split'] == 'test') & (df['embedding_type'] == 'DIFFUSION')]['original_img_path']
    eeg_embeddings = np.stack(df[(df['split'] == 'test') & (df['embedding_type'] == 'DIFFUSION')]['embeddings'])

    cosine_scores = [
        cosine_similarity(eeg[np.newaxis, :], target[np.newaxis, :])[0, 0]
        for eeg, target in zip(eeg_embeddings, target_embeddings)
    ]

    mse_scores = [
        mean_squared_error(eeg, target)
        for eeg, target in zip(eeg_embeddings, target_embeddings)
    ]

    lbp_scores, color_scores, reconstructed_paths = [], [], []
    for eeg_path, target_path in zip(eeg_paths, target_paths):
        sub_folder, img_folder = os.path.basename(os.path.dirname(target_path)), os.path.basename(target_path)
        reconstructed_path = os.path.join(
            base_path.replace('embeddings', 'generated_imgs').replace('EEG_encoder', ''),
            f'sub-{subject_id:02d}',
            sub_folder, img_folder
        )

        reconstructed_paths.append(reconstructed_path)

        original_img = cv2.imread(target_path)
        reconstructed_img = cv2.imread(reconstructed_path)

        if original_img is None or reconstructed_img is None:
            continue

        color_correlation = np.mean(list(compare_colors(original_img, reconstructed_img, 'corr').values()))
        color_scores.append(color_correlation)

        correlation_score = compare_textures(original_img, reconstructed_img)
        lbp_scores.append(correlation_score)

    return {
        'cosine_scores': cosine_scores,
        'mse_scores': mse_scores,
        'lbp_scores': lbp_scores,
        'color_scores': color_scores,
        'original_paths': target_paths,
        'reconstructed_paths': pd.Series(reconstructed_paths)
    }

def generate_metrics_table(base_path, save_path):
    """Generate a LaTeX table with metrics for all subjects."""
    all_metrics = []
    all_cosine_scores, all_mse_scores, all_lbp_scores, all_color_scores = [], [], [], []

    sem = lambda x: np.std(x, ddof=1) / np.sqrt(np.size(x))

    for subject_id in range(1, 11):
        metrics = calculate_subject_metrics(subject_id, base_path)
        
        # Collect all scores for overall statistics
        all_cosine_scores.extend(metrics['cosine_scores'])
        all_mse_scores.extend(metrics['mse_scores'])
        all_lbp_scores.extend(metrics['lbp_scores'])
        all_color_scores.extend(metrics['color_scores'])

        # Subject-level statistics
        subject_metrics = {
            'subject': f'sub-{subject_id:02d}',
            'cosine': np.mean(metrics['cosine_scores']),
            'cosine_std': sem(metrics['cosine_scores']),
            'mse': np.mean(metrics['mse_scores']),
            'mse_std': sem(metrics['mse_scores']),
            'lbp': np.mean(metrics['lbp_scores']),
            'lbp_std': sem(metrics['lbp_scores']),
            'color': np.mean(metrics['color_scores']),
            'color_std': sem(metrics['color_scores'])
        }
        all_metrics.append(subject_metrics)

    # Overall statistics across all instances
    overall_metrics = {
        'subject': 'Overall',
        'cosine': np.mean(all_cosine_scores),
        'cosine_std': sem(all_cosine_scores),
        'mse': np.mean(all_mse_scores),
        'mse_std': sem(all_mse_scores),
        'lbp': np.mean(all_lbp_scores),
        'lbp_std': sem(all_lbp_scores),
        'color': np.mean(all_color_scores),
        'color_std': sem(all_color_scores)
    }
    all_metrics.append(overall_metrics)

    metrics_df = pd.DataFrame(all_metrics)

    # Find the best subject for each metric
    best_cosine_subject = metrics_df.loc[metrics_df['cosine'].idxmax(), 'subject']
    best_mse_subject = metrics_df.loc[metrics_df['mse'].idxmin(), 'subject']
    best_lbp_subject = metrics_df.loc[metrics_df['lbp'].idxmax(), 'subject']
    best_color_subject = metrics_df.loc[metrics_df['color'].idxmax(), 'subject']

    latex_table = """\begin{table}[ht]
\centering
\begin{tabular}{lcccc}
\hline
Subject & Cosine Similarity & MSE & Color & LBP \\
\hline
"""

    for _, row in metrics_df.iterrows():
        color = ""
        if row['subject'] == best_cosine_subject:
            is_best_cosine = True
        else:
            is_best_cosine = False

        if row['subject'] == best_mse_subject:
            is_best_mse = True
        else:
            is_best_mse = False

        if row['subject'] == best_color_subject:
            is_best_color = True
        else:
            is_best_color = False

        if row['subject'] == best_lbp_subject:
            is_best_lbp = True
        else:
            is_best_lbp = False

        # Format values
        def format_value(mean, std, is_best):
            value = f"{mean:.4f} \\pm {std:.4f}"
            if is_best:
                return f"\\underline{{\\mathbf{{{value}}}}}"
            return f"${value}$"

        latex_table += (
            f"{color}{row['subject']} & "
            f"{format_value(row['cosine'], row['cosine_std'], is_best_cosine)} & "
            f"{format_value(row['mse'], row['mse_std'], is_best_mse)} & "
            f"{format_value(row['color'], row['color_std'], is_best_color)} & "
            f"{format_value(row['lbp'], row['lbp_std'], is_best_lbp)} \\"
        )

    latex_table += "\hline\end{tabular}\caption{Metrics by Subject}\label{tab:metrics}\end{table}"

    with open(save_path, 'w') as f:
        f.write(latex_table)
        
    print("Saved table")
    
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
import plotly.express as px


def plot_distributions(base_path, save_path=None):
    """Plot distributions for metrics across subjects."""
    metrics = {
        'cosine': {},
        'mse': {},
        'lbp': {},
        'color': {}
    }
    subjects = [f'sub-{i:02d}' for i in range(1, 11)]

    for subject_id in range(1, 11):
        subject_metrics = calculate_subject_metrics(subject_id, base_path)
        metrics['cosine'][f'sub-{subject_id:02d}'] = subject_metrics['cosine_scores']
        metrics['mse'][f'sub-{subject_id:02d}'] = subject_metrics['mse_scores']
        metrics['lbp'][f'sub-{subject_id:02d}'] = subject_metrics['lbp_scores']
        metrics['color'][f'sub-{subject_id:02d}'] = subject_metrics['color_scores']

    metric_names = ['Cosine Similarity', 'MSE', 'Color Correlation', 'LBP Correlation']
    metric_keys = ['cosine', 'mse', 'color', 'lbp']
    colors = px.colors.qualitative.Plotly

    for metric_name, metric_key in zip(metric_names, metric_keys):
        # Extract scores for the metric
        metric_scores = [metrics[metric_key][subject] for subject in subjects]

        # Calculate KDE for each subject
        xs = np.linspace(
            min(np.min(scores) for scores in metric_scores) - 0.1,
            max(np.max(scores) for scores in metric_scores) + 0.1,
            500
        )
        kdes = [gaussian_kde(scores, bw_method=0.2) for scores in metric_scores]

        # Normalize KDEs
        max_kde = max(kde(xs).max() for kde in kdes)
        overlap_factor = 1.9
        whiten_factor = 0.5

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 12))
        for idx, (subject, kde, color) in enumerate(zip(reversed(subjects), reversed(kdes), reversed(colors))):
            kde_values = kde(xs) / max_kde * overlap_factor
            ax.plot(xs, idx + kde_values, lw=2, color=color, zorder=50 - idx)

            # Fill under the curve
            whitened = np.array(to_rgb(color)) * (1 - whiten_factor) + whiten_factor
            ax.fill_between(xs, idx, idx + kde_values, color=whitened, alpha=0.8, zorder=50 - idx)

        # Customize the plot
        ax.set_xlim(xs[0], xs[-1])
        ax.set_xlabel(f'Distribution of {metric_name}', fontsize=14)
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels(list(reversed(subjects)), fontsize=12)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/{metric_key}_distribution.png", dpi=300)
        else:
            plt.show()

def plot_top_and_bottom_recreations(subject_id, metric, base_path, save_path=None):
    """Plot the top and bottom 10 recreations for a subject within a metric."""
    metrics = calculate_subject_metrics(subject_id, base_path)

    metric_scores = metrics[f"{metric}_scores"]
    original_paths = metrics['original_paths']
    reconstructed_paths = metrics['reconstructed_paths']

    all_scores = {
        'Cosine': metrics['cosine_scores'],
        'MSE': metrics['mse_scores'],
        'LBP': metrics['lbp_scores'],
        'Color': metrics['color_scores']
    }

    # Sort indices based on scores (for MSE, lower is better)
    if metric == 'mse':
        top_indices = np.argsort(metric_scores)[:10]
        bottom_indices = np.argsort(metric_scores)[-10:][::-1]
    else:
        top_indices = np.argsort(metric_scores)[-10:][::-1]
        bottom_indices = np.argsort(metric_scores)[:10]

    # Prepare figure with no spacing between images in "Best" and "Worst" sections
    fig, axes = plt.subplots(5, 10, figsize=(20, 10), gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [1, 1, 0.25, 1, 1]})

    def add_row(images, row_index, indices, show_scores=False):
        """Add a row of images to the plot."""
        for col, img_index in enumerate(indices):
            img_path = images.iloc[img_index]  # Use .iloc for positional indexing
            img = cv2.imread(img_path)[..., ::-1]  # Convert BGR to RGB
            axes[row_index, col].imshow(img)
            axes[row_index, col].axis('off')
            # Add scores below the images if required
            if show_scores:
                score_text = (
                    f"CS: {all_scores['Cosine'][img_index]:.2f}, MSE: {all_scores['MSE'][img_index]:.2f}\n"
                    f"Col: {all_scores['Color'][img_index]:.2f}, LBP: {all_scores['LBP'][img_index]:.2f}"
                )
                axes[row_index, col].text(
                    0.5, -0.05, score_text, fontsize=12, ha='center', va='top', transform=axes[row_index, col].transAxes
                )

    # Add "Best" recreations
    add_row(original_paths, 0, top_indices)  # Originals, no scores
    add_row(reconstructed_paths, 1, top_indices, show_scores=True)  # Reconstructed with scores

    # Add vertical spacing by leaving the third row blank
    for ax in axes[2, :]:
        ax.axis('off')

    # Add "Worst" recreations
    add_row(original_paths, 3, bottom_indices)  # Originals, no scores
    add_row(reconstructed_paths, 4, bottom_indices, show_scores=True)  # Reconstructed with scores

    # Add labels for "Best" and "Worst" sections
    fig.text(0.0075, 24.5/32, "Best", rotation=90, va='center', ha='center', fontsize=20, fontweight='bold')
    fig.text(0.0075, 9.5/32, "Worst", rotation=90, va='center', ha='center', fontsize=20, fontweight='bold')
    fig.text(0.0175, 28/32, "Seen", rotation=90, va='center', ha='center', fontsize=16)
    fig.text(0.0175, 21/32, "Generated", rotation=90, va='center', ha='center', fontsize=16)
    fig.text(0.0175, 13/32, "Seen", rotation=90, va='center', ha='center', fontsize=16)
    fig.text(0.0175, 6/32, "Generated", rotation=90, va='center', ha='center', fontsize=16)

    plt.tight_layout(rect=[0.02, 0, 1, 1])  # Adjust 'left' value as needed

    # Save or display the plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"subject_{subject_id:02d}_{metric}_top_bottom.png"), dpi=300)

        print("saved plot...!")
    else:
        plt.show()

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def plot_recreations_global(subjects, metrics, base_path, save_path=None, top=True, dpi=100, num_images=50):
    """
    Plot the best or worst recreations across all subjects, based on a weighted combination of multiple metrics.
    
    Args:
        subjects (list): List of subject IDs to include (e.g., [1, 2, 3]).
        metrics (list): List of metrics to use for ranking (e.g., ['cosine', 'mse']).
        base_path (str): Path to the embeddings directory.
        save_path (str): Path to save the plot. If None, display the plot.
        top (bool): If True, plot the best recreations; if False, plot the worst.
        dpi (int): Dots per inch for the figure, used for sizing.
        num_images (int): Total number of images to display (multiple plots if >10).
    """
    best_original_paths = []
    best_reconstructed_paths = []
    all_metrics = {metric: [] for metric in metrics}
    
    for subject_id in subjects:
        metrics_data = calculate_subject_metrics(subject_id, base_path)
        original_paths = metrics_data["original_paths"]
        reconstructed_paths = metrics_data["reconstructed_paths"]
        
        for metric in metrics:
            all_metrics[metric].extend(np.array(metrics_data[f"{metric}_scores"]))
        
        best_original_paths.extend(original_paths.tolist())
        best_reconstructed_paths.extend(reconstructed_paths.tolist())
    
    # Normalize each metric
    normalized_metrics = {}
    for metric in metrics:
        values = np.array(all_metrics[metric])
        if metric == 'mse':  # Lower is better
            values = 1 - (values - values.min()) / (values.max() - values.min())
        else:  # Higher is better
            values = (values - values.min()) / (values.max() - values.min())
        normalized_metrics[metric] = values
    
    # Compute combined score
    weights = np.ones(len(metrics)) / len(metrics)  # Equal weights
    combined_scores = sum(weights[i] * normalized_metrics[metric] for i, metric in enumerate(metrics))
    
    # Sort based on combined score
    best_indices = np.argsort(combined_scores)[-num_images:][::-1] if top else np.argsort(combined_scores)[:num_images]
    best_original_paths = [best_original_paths[i] for i in best_indices]
    best_reconstructed_paths = [best_reconstructed_paths[i] for i in best_indices]
    sorted_metrics = {metric: [all_metrics[metric][i] for i in best_indices] for metric in metrics}
    
    total_height_units = sum([1, 1, 0.25])  # Two image rows and one text row
    fig_width = 500 * 10 / dpi  # Fixed width for 10 columns
    fig_height = 500 * total_height_units / dpi  # Dynamic height calculation
    
    # Generate multiple plots if needed
    for start in range(0, num_images, 10):
        end = min(start + 10, num_images)
        fig, axes = plt.subplots(3, 10, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': [1, 1, 0.25]})
        
        def add_images(images, row_index):
            for col, img_path in enumerate(images):
                img = cv2.imread(img_path)[..., ::-1]
                axes[row_index, col].imshow(img)
                axes[row_index, col].axis("off")
        
        add_images(best_original_paths[start:end], 0)
        add_images(best_reconstructed_paths[start:end], 1)
        
        # Add metric label to the left
        fig.text(0.0075, 0.5, " & ".join(metrics), rotation=90, va="center", ha="center", fontsize=40, fontweight="bold")
        fig.text(0.0175, 0.75, "Seen", rotation=90, va="center", ha="center", fontsize=32)
        fig.text(0.0175, 0.25, "Generated", rotation=90, va="center", ha="center", fontsize=32)
        
        # Add all scores under each image
        for col in range(end - start):
            score_text = "\n".join([f"{metric}: {sorted_metrics[metric][start + col]:.2f}" for metric in metrics])
            axes[2, col].text(0.5, 0.5, score_text, fontsize=24, ha="center", va="center")
            axes[2, col].axis("off")
        
        # Adjust layout
        plt.tight_layout(rect=[0.03, 0, 1, 1])
        
        # Save or display
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            suffix = "top" if top else "worst"
            plt.savefig(os.path.join(save_path, f"{suffix}_{'_'.join(metrics)}_recreations_{start+1}-{end}.png"), dpi=dpi)
            print(f"Saved plot: {suffix}_{'_'.join(metrics)}_recreations_{start+1}-{end}.png")
        else:
            plt.show()

         
def plot_recreations_by_subjects(subjects, metric, base_path, save_path=None, top=True, dpi=100):
    """
    Plot the best or worst recreations for multiple subjects.
    
    Args:
        subjects (list): List of subject IDs to include (e.g., [1, 2, 3]).
        metric (str): Metric to sort by ('cosine', 'mse', 'lbp', 'color').
        base_path (str): Path to the embeddings directory.
        save_path (str): Path to save the plot. If None, display the plot.
        top (bool): If True, plot the best recreations; if False, plot the worst.
        dpi (int): Dots per inch for the figure, used for sizing.
    """
    all_metrics = []
    all_original_paths = []
    all_reconstructed_paths = []

    for subject_id in subjects:
        metrics = calculate_subject_metrics(subject_id, base_path)

        metric_scores = metrics[f"{metric}_scores"]
        original_paths = metrics["original_paths"]
        reconstructed_paths = metrics["reconstructed_paths"]

        if metric == 'mse':
            # For MSE: lower is better
            if top:
                indices = np.argsort(metric_scores)[:10]  # Lowest MSEs for Top
            else:
                indices = np.argsort(metric_scores)[-10:][::-1]  # Highest MSEs for Worst
        else:
            # For other metrics: higher is better
            if top:
                indices = np.argsort(metric_scores)[-10:][::-1]  # Highest scores for Top
            else:
                indices = np.argsort(metric_scores)[:10]  # Lowest scores for Worst

        all_metrics.append({key: np.array(metrics[key])[indices] for key in metrics if "scores" in key})
        all_original_paths.append(original_paths.iloc[indices])
        all_reconstructed_paths.append(reconstructed_paths.iloc[indices])

    # Calculate height ratios
    num_subjects = len(subjects)
    height_ratios = []
    for _ in range(num_subjects):
        height_ratios.extend([1, 1, 0.25])  # Two full rows for images, one smaller row for spacing
    height_ratios.pop()  # Remove the last spacing row

    # Calculate total figure height in inches
    fig_width = 500 * 10 / dpi  # 10 images per row, each 500 pixels wide
    total_height_units = sum(height_ratios)  # Total height units from height_ratios
    fig_height = 500 * total_height_units / dpi  # Convert to inches based on dpi

    fig, axes = plt.subplots(
        len(height_ratios), 10, figsize=(fig_width, fig_height),
        gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': height_ratios}
    )

    def add_subject_row(images, row_index, metrics=None, show_scores=False):
        for col, img_path in enumerate(images):
            img = cv2.imread(img_path)[..., ::-1]
            axes[row_index, col].imshow(img)
            axes[row_index, col].axis("off")
            if show_scores and metrics:
                score_text = (
                    f"CS: {metrics['cosine_scores'][col]:.2f}, MSE: {metrics['mse_scores'][col]:.2f}\n"
                    f"Col: {metrics['color_scores'][col]:.2f}, LBP: {metrics['lbp_scores'][col]:.2f}"
                )
                axes[row_index, col].text(
                    0.5, -0.05, score_text, fontsize=24, ha="center", va="top",
                    transform=axes[row_index, col].transAxes
                )

    for i, subject_id in enumerate(subjects):
        base_row = i * 3  # Each subject occupies 3 rows: 2 image rows + 1 spacing row
        add_subject_row(all_original_paths[i], base_row)
        add_subject_row(all_reconstructed_paths[i], base_row + 1, metrics=all_metrics[i], show_scores=True)

        # Add subject label
        fig.text(0.0075, (sum(height_ratios) - base_row - 1) / sum(height_ratios), f"sub-{subject_id:02d}",
                 rotation=90, va="center", ha="center", fontsize=40, fontweight="bold")

        # Add "Seen" and "Generated" labels
        fig.text(0.0175, (sum(height_ratios) - base_row - 0.5) / sum(height_ratios), "Seen",
                 rotation=90, va="center", ha="center", fontsize=32)
        fig.text(0.0175, (sum(height_ratios) - base_row - 1.5) / sum(height_ratios), "Generated",
                 rotation=90, va="center", ha="center", fontsize=32)

        # Remove axes from the spacing row
        if base_row + 2 < len(height_ratios):
            for ax in axes[base_row + 2, :]:
                ax.axis("off")

    # Adjust layout for extra space on the left
    plt.tight_layout(rect=[0.03, 0, 1, 1])

    # Save or display the plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        suffix = "top" if top else "worst"
        plt.savefig(os.path.join(save_path, f"{suffix}_{metric}_recreations_subjects_{'_'.join([str(sub) for sub in subjects])}.png"), dpi=dpi)
        print("Saved plot!")
    else:
        plt.show()

# Example call
# plot_top_and_bottom_recreations(subject_id=1, metric='cosine', base_path="/path/to/data", save_path="./plots")

        

def plot_topk_bottomk_results(base_path, split_type='test', target_type='IMG', trained_type='ATMS', k=5, save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    mse_results = []

    # Loop over all subjects
    for subject_id in range(1, 11):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings = load_embeddings(subject_folder)  # Assuming load_embeddings is defined
        df_split = embeddings[embeddings['split'] == split_type]

        for img_path in df_split['original_img_path'].unique():
            subset = df_split[df_split['original_img_path'] == img_path]
            target_embeddings = subset[subset['embedding_type'] == target_type]['embeddings'].values
            trained_embeddings = subset[subset['embedding_type'] == trained_type]['embeddings'].values

            if len(target_embeddings) == 1 and len(trained_embeddings) == 1:
                target = np.array(target_embeddings[0])
                trained = np.array(trained_embeddings[0])

                mse = mean_squared_error(target, trained)
                cosine_sim = cosine_similarity([target], [trained])[0][0]

                mse_results.append({
                    'Subject': f'sub-{subject_id:02d}',
                    'MSE': mse,
                    'Cosine Similarity': cosine_sim,
                    'Image Path': img_path
                })

    mse_df = pd.DataFrame(mse_results)

    # Separate top-k and bottom-k results for each metric
    for metric, ascending in [('MSE', True), ('Cosine Similarity', False)]:
        for rank_type in ['top', 'bottom']:
            fig, axes = plt.subplots(10, k, figsize=(k * 3, 30))
            metric_title = f"{rank_type.capitalize()}-{k} {metric}"

            for idx, subject in enumerate(mse_df['Subject'].unique()):
                subject_data = mse_df[mse_df['Subject'] == subject]
                if rank_type == 'top':
                    ranked_data = subject_data.nsmallest(k, metric) if ascending else subject_data.nlargest(k, metric)
                else:
                    ranked_data = subject_data.nlargest(k, metric) if ascending else subject_data.nsmallest(k, metric)

                for i, img_path in enumerate(ranked_data['Image Path']):
                    if i < k:  # Ensure we only plot up to k images
                        img = imread(img_path)
                        ax = axes[idx, i]
                        ax.imshow(img)
                        ax.axis('off')
                        if idx == 0:
                            ax.set_title(f"Image {i+1}")
                axes[idx, 0].set_ylabel(subject, rotation=0, labelpad=50, fontsize=12)

            plt.suptitle(metric_title, fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            save_path = os.path.join(save_dir, f"{metric_title.replace(' ', '_')}.png")
            plt.savefig(save_path)
            plt.close()

def write_category_accuracy_to_latex(base_path, target_type='IMG', trained_type='ATMS', save_path='category_scores_table.txt', seed=42):
    """
    Calculate overall accuracy, category accuracy, and per-category accuracy, and save them as a formatted LaTeX table.
    Includes mean and confidence intervals for each score (based on images, not across subjects).
    Adds an overall average row across subjects for each column.
    """
    random.seed(seed)

    

    def load_subject_embeddings(subject_id):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        return embeddings_df[embeddings_df['split'] == 'test']

    def calculate_confidence_interval(success_count, total_count):
        if total_count == 0:
            return 0, 0
        proportion = success_count / total_count
        ci = 1.96 * np.sqrt((proportion * (1 - proportion)) / total_count)
        return proportion, ci

    def calculate_metrics(subject_id):
        embeddings_df = load_subject_embeddings(subject_id)

        target_embeddings = (
            embeddings_df[embeddings_df['embedding_type'] == target_type]
            .sort_values(by='original_img_path')
            .head(200)
        )
        trained_embeddings = (
            embeddings_df[embeddings_df['embedding_type'] == trained_type]
            .sort_values(by='original_img_path')
            .head(200)
        )

        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        similarity_matrix = cosine_similarity(trained_matrix, target_matrix)

        correct_filepaths = list(target_embeddings['original_img_path'])
        target_categories = target_embeddings['category'].values

        total_correct = 0
        total_instances = len(correct_filepaths)

        per_category_counts = {category: {'correct': 0, 'total': 0} for category in np.unique(target_categories)}
        total_category_correct = 0

        for i, row in enumerate(similarity_matrix):
            predicted_idx = np.argmax(row)
            predicted_filepath = correct_filepaths[predicted_idx]
            predicted_category = target_categories[predicted_idx]

            if correct_filepaths[i] == predicted_filepath:
                total_correct += 1

            current_category = target_categories[i]
            per_category_counts[current_category]['total'] += 1
            if target_categories[i] == predicted_category:
                total_category_correct += 1
                per_category_counts[current_category]['correct'] += 1

        overall_accuracy = (total_correct, total_instances)
        overall_category_accuracy = (total_category_correct, total_instances)

        return {
            'overall_accuracy': overall_accuracy,
            'overall_category_accuracy': overall_category_accuracy,
            'per_category_counts': per_category_counts
        }

    subjects = [f'sub-{i:02d}' for i in range(1, 11)]
    total_correct = 0
    total_instances = 0
    total_category_correct = 0
    total_category_counts = {}

    all_results = []

    for subject_id in range(1, 11):
        metrics = calculate_metrics(subject_id)

        total_correct += metrics['overall_accuracy'][0]
        total_instances += metrics['overall_accuracy'][1]
        total_category_correct += metrics['overall_category_accuracy'][0]

        for category, counts in metrics['per_category_counts'].items():
            if category not in total_category_counts:
                total_category_counts[category] = {'correct': 0, 'total': 0}
            total_category_counts[category]['correct'] += counts['correct']
            total_category_counts[category]['total'] += counts['total']

        subject_results = {
            'Subject': f'sub-{subject_id:02d}',
            'Accuracy': calculate_confidence_interval(
                metrics['overall_accuracy'][0], metrics['overall_accuracy'][1]
            ),
            'Category Accuracy': calculate_confidence_interval(
                metrics['overall_category_accuracy'][0], metrics['overall_category_accuracy'][1]
            )
        }

        subject_results.update({
            category: calculate_confidence_interval(counts['correct'], counts['total'])
            for category, counts in metrics['per_category_counts'].items()
        })

        all_results.append(subject_results)

    overall_accuracy = calculate_confidence_interval(total_correct, total_instances)
    overall_category_accuracy = calculate_confidence_interval(total_category_correct, total_instances)
    overall_per_category = {
        category: calculate_confidence_interval(
            counts['correct'], counts['total']
        )
        for category, counts in total_category_counts.items()
    }

    overall_row = {
        'Subject': 'Overall Average',
        'Accuracy': overall_accuracy,
        'Category Accuracy': overall_category_accuracy
    }
    overall_row.update(overall_per_category)
    all_results.append(overall_row)

    df = pd.DataFrame(all_results)

    def format_value(value):
        mean, ci = value
        return f"{mean:.3f} \\pm {ci:.3f}"

    df = df.applymap(lambda x: format_value(x) if isinstance(x, tuple) else x)

    bolded_cols = {
        col: df.iloc[:-1][col].apply(lambda x: float(x.split(' \\pm ')[0])).idxmax()
        for col in df.columns if col != 'Subject'
    }

    latex_string = "\\begin{table}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{\n"
    latex_string += "\\begin{tabular}{l|c" + "c" * (len(df.columns) - 2) + "}\n"
    latex_string += "\\hline\\hline\n"
    latex_string += "\\textbf{Subject} & \\textbf{Accuracy} & \\textbf{Category Accuracy} & "
    latex_string += " & ".join([f"\\textbf{{{col}}}" for col in df.columns if col not in ['Subject', 'Accuracy', 'Category Accuracy']]) + " \\\\ \\hline\n"

    for i, row in df.iterrows():
        row_str = ""
        if row['Subject'] == 'Overall Average':
            row_str += "\\rowcolor{gray!20}"
        elif i == bolded_cols['Accuracy']:
            row_str += "\\rowcolor{blue!20}"

        for col in df.columns:
            if col == 'Subject':
                row_str += f"\\textbf{{{row[col]}}} & "
            else:
                value = row[col]
                if i == bolded_cols[col]:
                    value = f"\\underline{{$\\mathbf{{{value}}}$}}"
                else:
                    value = f"${value}$"
                row_str += f"{value} & "

        latex_string += row_str[:-2] + "\\\\ \\hline\n"

    latex_string += "\\end{tabular}\n}\n"
    latex_string += "\\caption{Overall accuracy, category accuracy, and per-category accuracy with confidence intervals.}\n"
    latex_string += "\\label{tab:category_accuracy_metrics}\n\\end{table}"

    with open(save_path, 'w') as f:
        f.write(latex_string)

    print(f"LaTeX table saved to {save_path}")







def write_kway_to_latex(base_path, target_type='IMG', trained_type='ATMS', save_path='scores_table.txt', seed=42):
    """
    Calculate accuracy, top-5 accuracy, and k-way scores, and save them as a formatted LaTeX table.
    A seed is used to ensure reproducibility of K-way score calculations.
    Confidence intervals are used instead of standard deviations.
    """
    random.seed(seed)

    sem = lambda x: np.std(x, ddof=1) / np.sqrt(np.size(x))

    def load_subject_embeddings(subject_id):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        return embeddings_df[embeddings_df['split'] == 'test']

    def calculate_metrics(subject_id):
        embeddings_df = load_subject_embeddings(subject_id)
        
        target_embeddings = (
            embeddings_df[embeddings_df['embedding_type'] == target_type]
            .sort_values(by='original_img_path')
            .head(200)
        )
        trained_embeddings = (
            embeddings_df[embeddings_df['embedding_type'] == trained_type]
            .sort_values(by='original_img_path')
            .head(200)
        )
        
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        similarity_matrix = cosine_similarity(trained_matrix, target_matrix)
        
        correct_filepaths = list(target_embeddings['original_img_path'])
        
        accuracy = 0
        top5_accuracy = 0
        kway_scores = {2: 0, 4: 0, 10: 0}
        total = len(correct_filepaths)

        def calculate_confidence_interval(success_count):
            proportion = success_count / total
            ci = 1.96 * np.sqrt((proportion * (1 - proportion)) / total)
            return proportion, ci
        
        for i, row in enumerate(similarity_matrix):
            sorted_indices = np.argsort(-row)
            sorted_filepaths = [correct_filepaths[idx] for idx in sorted_indices]
            
            if correct_filepaths[i] == sorted_filepaths[0]:
                accuracy += 1
            if correct_filepaths[i] in sorted_filepaths[:5]:
                top5_accuracy += 1

            for k in kway_scores.keys():
                sampled_indices = random.sample([idx for idx in range(len(correct_filepaths)) if idx != i], k - 1)
                sampled_filepaths = [correct_filepaths[idx] for idx in sampled_indices] + [correct_filepaths[i]]
                sampled_similarities = [row[idx] for idx in sampled_indices] + [row[i]]
                
                if np.argmax(sampled_similarities) == len(sampled_similarities) - 1:
                    kway_scores[k] += 1

        return {
            'accuracy': calculate_confidence_interval(accuracy),
            'top5_accuracy': calculate_confidence_interval(top5_accuracy),
            **{f'{k}-way': calculate_confidence_interval(kway_scores[k]) for k in kway_scores}
        }

    subjects = [f'sub-{i:02d}' for i in range(1, 11)]
    results = []
    all_scores = {'accuracy': [], 'top5_accuracy': [], '2-way': [], '4-way': [], '10-way': []}

    for subject_id in range(1, 11):
        scores = calculate_metrics(subject_id)
        results.append({
            'Subject': f'sub-{subject_id:02d}',
            'Accuracy': scores['accuracy'],
            'Top-5': scores['top5_accuracy'],
            '2-Way': scores['2-way'],
            '4-Way': scores['4-way'],
            '10-Way': scores['10-way']
        })

        all_scores['accuracy'].extend([scores['accuracy'][0]] * 200)
        all_scores['top5_accuracy'].extend([scores['top5_accuracy'][0]] * 200)
        all_scores['2-way'].extend([scores['2-way'][0]] * 200)
        all_scores['4-way'].extend([scores['4-way'][0]] * 200)
        all_scores['10-way'].extend([scores['10-way'][0]] * 200)

    df = pd.DataFrame(results)

    # Calculate overall averages and confidence intervals
    overall_scores = {
        metric: (
            np.mean(all_scores[metric]),
            1.96 * np.sqrt(np.var(all_scores[metric]) / len(all_scores[metric]))
        )
        for metric in all_scores
    }

    # Identify the best scores for bolding
    bolded_cols = {col: df[col].apply(lambda x: x[0]).idxmax() for col in df.columns if col != 'Subject'}

    # Build the LaTeX table string
    latex_string = "\\begin{table}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{ % Resizes the table to fit within the text width\n"
    latex_string += "\\begin{tabular}{l|cccccc}\n"
    latex_string += "\\hline\\hline\n"
    latex_string += "\\textbf{Subject} & \\textbf{Accuracy} & \\textbf{Top-5} & \\textbf{2-Way} & \\textbf{4-Way} & \\textbf{10-Way} \\\\ \\hline\n"

    for i, row in df.iterrows():
        row_str = ""
        if i == bolded_cols['Accuracy']:
            row_str += "\\rowcolor{blue!20}"

        for col in df.columns:
            if col == 'Subject':
                row_str += f"\\textbf{{{row[col]}}} & "
            else:
                mean, ci = row[col]
                if i == bolded_cols[col]:
                    value = f"\\underline{{$\\mathbf{{{mean:.3f} \\pm {ci:.3f}}}$}}"
                else:
                    value = f"${mean:.3f} \\pm {ci:.3f}$"
                row_str += f"{value} & "

        latex_string += row_str[:-2] + "\\\\ \\hline\n"

    # Add row for overall averages
    latex_string += "\\textbf{Overall Average} & "
    for metric in ['accuracy', 'top5_accuracy', '2-way', '4-way', '10-way']:
        mean, ci = overall_scores[metric]
        latex_string += f"${mean:.3f} \\pm {ci:.3f}$ & "
    latex_string = latex_string[:-2] + "\\\\ \\hline\n"

    latex_string += "\\end{tabular}\n}\n"
    latex_string += "\\caption{Performance metrics for subjects. The highlighted values indicate the best results for each metric. The last row shows the overall averages.}\n"
    latex_string += "\\label{tab:performance_metrics}\n\\end{table}"

    # Save the LaTeX table to a file
    with open(save_path, 'w') as f:
        f.write(latex_string)

    print(f"LaTeX table saved to {save_path}")








def load_embeddings(folder_path):
    # List to store dataframes from each file
    dfs = []

    # Loop through each .h5 file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder_path, filename)

            # Open each HDF5 file and load data
            with h5py.File(file_path, 'r') as h5f:
                if len(h5f.keys()) == 0:
                    continue
                else:
                    embedding_key = [key for key in h5f.keys() if key not in {"original_img_path", "class_name", "split"}][0]
                    embeddings = h5f[embedding_key][:]
                    img_paths = h5f["original_img_path"][:]
                    class_names = h5f["class_name"][:]
                    split_labels = h5f["split"][:]

                    # Convert the data into a pandas DataFrame
                    df = pd.DataFrame({
                        "original_img_path": img_paths,
                        "class_name": class_names,
                        "split": split_labels
                    })

                    # Convert embeddings to list and add as a column in the DataFrame
                    df["embeddings"] = list(embeddings)

                    # Add the filename (excluding .h5) as a new column
                    df["embedding_type"] = filename[:-14].upper()  # Remove .h5 extension

                    # Append the dataframe to the list
                    dfs.append(df)

    # Concatenate all dataframes in the list
    final_df = pd.concat(dfs, ignore_index=True)

    # Decode any bytes to strings
    for col in ['original_img_path', 'class_name', 'split']:
        final_df[col] = final_df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)


    # Rename classnames and get categories
    import pickle

    # Load dictionary from a file
    def load_dict_pickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    class2duperclass = load_dict_pickle('/work3/s184984/repos/EEG_Image_decode/Generation/super_classes/class2duperclass.pkl')

    remove_digits = lambda u: ''.join(filter(lambda x: not x.isdigit(), u))

    final_df['category'] = final_df['class_name'].apply(lambda x: class2duperclass[remove_digits(x)])

    return final_df



from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm

# Function to calculate most similar targets in batches
def calculate_most_similar_in_batches(diffusion_embeddings, target_embeddings, batch_size=100):
    """
    Find the indices of the most similar targets in batches.
    
    Parameters:
        diffusion_embeddings: np.ndarray
            The embeddings for diffusion data.
        target_embeddings: np.ndarray
            The embeddings for target data.
        batch_size: int
            The batch size to use for processing.
    
    Returns:
        np.ndarray: Indices of the most similar targets.
    """
    most_similar_indices = []
    
    for start_idx in tqdm(range(0, diffusion_embeddings.shape[0], batch_size), desc='Processing Batches'):
        end_idx = min(start_idx + batch_size, diffusion_embeddings.shape[0])
        batch = diffusion_embeddings[start_idx:end_idx]
        # Compute cosine similarity for the current batch
        batch_similarities = cosine_similarity(batch, target_embeddings)
        # Find the most similar targets for each item in the batch
        batch_most_similar = np.argmax(batch_similarities, axis=1)
        most_similar_indices.extend(batch_most_similar)
    
    return np.array(most_similar_indices)


import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.io as pio

def plot_mse_cosine_similarity_all_subjects(base_path, split_type='test', target_type='IMG', trained_type='ATMS',
                                            color_by='Subject',
                                            save_path='/work3/s184984/repos/EEG_Image_decode/Generation/results/scores/', 
                                            file_name='MSEvsCosine.png',
                                            width=1000, height=400, scale=None):
    
    # Prepare lists to store results
    mse_list = []
    cosine_sim_list = []
    img_paths = []
    subjects = []
    categories = []
    
    # Loop over all subject folders
    for subject_id in range(1, 11):  # Assuming sub-01 to sub-10
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        print(f'Processing {subject_folder}...')
        
        # Load embeddings for the subject
        embeddings = load_embeddings(subject_folder)  # Assuming load_embeddings is defined elsewhere
        
        # Filter dataframe based on split type
        df_split = embeddings[embeddings['split'] == split_type]
        
        # Group by 'original_img_path' to find matching pairs
        for img_path in df_split['original_img_path'].unique():
            subset = df_split[df_split['original_img_path'] == img_path]
            
            # Extract target and trained embeddings based on the specified types
            target_embeddings = subset[subset['embedding_type'] == target_type]['embeddings'].values
            trained_embeddings = subset[subset['embedding_type'] == trained_type]['embeddings'].values
            
            # Ensure we have exactly one target and one trained embedding
            if len(target_embeddings) == 1 and len(trained_embeddings) == 1:
                target = np.array(target_embeddings[0])
                trained = np.array(trained_embeddings[0])
                
                # Calculate MSE
                mse = mean_squared_error(target, trained)
                mse_list.append(mse)
                
                # Calculate Cosine Similarity
                cosine_sim = cosine_similarity([target], [trained])[0][0]
                cosine_sim_list.append(cosine_sim)

                categories.append(list(subset['category'])[0])
                
                # Save image path and subject for tracking
                img_paths.append(img_path)
                subjects.append(f'sub-{subject_id:02d}')
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'MSE': mse_list,
        'Cosine Similarity': cosine_sim_list,
        'Image Path': img_paths,
        'Subject': subjects,
        'Category': categories
    })
    
    # Specify a color scale with sufficient distinct colors
    custom_colors = px.colors.qualitative.Light24  # Or px.colors.qualitative.Set3, Set1, Dark2, etc.

    # Create a scatter plot using Plotly
    fig = px.scatter(
        plot_df,
        x='MSE',
        y='Cosine Similarity',
        color=color_by,
        hover_data=['Image Path'],
        title=None,
        labels={'MSE': 'Mean Squared Error', 'Cosine Similarity': 'Cosine Similarity'},
        color_discrete_sequence=custom_colors  # Use the custom color sequence
    )

    # Update layout
    fig.update_layout(
        font=dict(
            family="Editorial New Italic",
            size=14
        )
    )

    # Save the plot to a PNG file
    if save_path and file_name:
        pio.write_image(fig, save_path + file_name, width=width, height=height, scale=scale)
    else:
        fig.show()



import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_pca_embeddings_specifically_for_split(base_path, target_type='IMG', trained_type='ATMS', 
                        color_by='Split', include_all_subjects=True, 
                        subject_id=None, include_target=True, 
                        save_path='/work3/s184984/repos/EEG_Image_decode/Generation/results/PCA', 
                        filename='pca_plot.png', width=1500, height=500, scale=None):
    # Step 1: Load target embeddings once and compute PCA
    print("Loading and computing PCA for target embeddings...")
    target_folder = os.path.join(base_path, 'sub-01')  # Load from any subject as target embeddings are the same
    target_embeddings = load_embeddings(target_folder)
    
    target_data = target_embeddings[target_embeddings['embedding_type'] == target_type]
    target_embedding_values = np.array(target_data['embeddings'].tolist())
    
    # Perform PCA on target embeddings with all components
    pca_full = PCA()  # Keep all components to compute the full variance
    pca_full.fit(target_embedding_values)
    cumulative_explained_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100  # Cumulative variance
    
    # Plot cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('Cumulative Explained Variance by PCA Components')
    cumulative_save_path = os.path.join(save_path, target_type, 'cumulative_explained_variance.png')
    os.makedirs(os.path.dirname(cumulative_save_path), exist_ok=True)
    plt.savefig(cumulative_save_path)
    plt.close()
    print(f"Cumulative Explained Variance plot saved to {cumulative_save_path}")
    
    # Reduce to 2 components for the PCA plot
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(target_embedding_values)
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
    
    # Save target PCA results
    target_pca_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Image Path': target_data['original_img_path'],
        'Category': target_data['category'],
        'Split': target_data['split'],
        'Subject': 'Target',
        'Embedding Type': target_type
    })
    
    # Step 2: Process trained embeddings subject by subject
    all_pca_results = [] if not include_target else [target_pca_df]  # Include target PCA results if specified
    if include_all_subjects:
        subjects_to_process = range(1, 11)
    elif subject_id:
        subjects_to_process = [subject_id]
    else:
        subjects_to_process = []  # No subjects, just target
    
    for subj_id in subjects_to_process:
        subject_folder = os.path.join(base_path, f'sub-{subj_id:02d}')
        print(f'Processing trained embeddings for {subject_folder}...')
        
        # Load trained embeddings
        trained_embeddings = load_embeddings(subject_folder)
        trained_data = trained_embeddings[trained_embeddings['embedding_type'] == trained_type]
        trained_embedding_values = np.array(trained_data['embeddings'].tolist())
        
        # Project trained embeddings onto PCA space
        trained_pca_result = pca.transform(trained_embedding_values)
        
        # Save trained PCA results
        trained_pca_df = pd.DataFrame({
            'PCA1': trained_pca_result[:, 0],
            'PCA2': trained_pca_result[:, 1],
            'Image Path': trained_data['original_img_path'],
            'Category': trained_data['category'],
            'Split': trained_data['split'],
            'Subject': f'sub-{subj_id:02d}',
            'Embedding Type': trained_type
        })
        
        all_pca_results.append(trained_pca_df)
    
    # Combine all PCA results
    combined_pca_df = pd.concat(all_pca_results, ignore_index=True)
    
    # Ensure test points are plotted last
    if 'Split' in combined_pca_df.columns:
        combined_pca_df['Split'] = pd.Categorical(combined_pca_df['Split'], categories=['train', 'test'], ordered=True)
    
    # Step 3: Create the PCA plot
    fig = px.scatter(
        combined_pca_df,
        x='PCA1',
        y='PCA2',
        color=color_by,
        symbol='Embedding Type',
        hover_data=['Image Path', 'Subject', 'Split', 'Category'],
        title=f'PCA of Target and Trained Embeddings',
        labels={
            'PCA1': f'PCA1 (EV: {explained_variance[0]:.2f}%)',
            'PCA2': f'PCA2 (EV: {explained_variance[1]:.2f}%)'
        },
        color_discrete_map={'train': 'blue', 'test': 'red', 'Target': 'green'}  # Fixed colors
    )
    
    # Reorder traces to ensure test is plotted last
    test_trace = None
    other_traces = []
    for trace in fig.data:
        if 'test' in trace.name.lower():
            test_trace = trace
        else:
            other_traces.append(trace)
    
    # If test trace exists, re-add it last
    if test_trace:
        fig.data = tuple(other_traces + [test_trace])
    
    # Update font style and layout
    fig.update_layout(
        font=dict(family="Editorial New Italic", size=16),
        width=width,
        height=height
    )
    
    # Optionally save the plot
    if save_path and filename:
        save_file_path = os.path.join(save_path, target_type, filename)
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        pio.write_image(fig, save_file_path, width=width, height=height, scale=scale)
        print(f"Plot saved to {save_file_path}")

    return fig  # Return the figure


def plot_pca_embeddings(base_path, target_type='IMG', trained_type='ATMS', 
                        color_by='Split', include_all_subjects=True, 
                        subject_id=None, include_target=True, 
                        save_path='/work3/s184984/repos/EEG_Image_decode/Generation/results/PCA', 
                        filename='pca_plot.png', width=1500, height=500, scale=None):
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    import os
    import plotly.express as px
    import plotly.io as pio
    import matplotlib.pyplot as plt
    
    # Step 1: Load target embeddings once and compute PCA
    print("Loading and computing PCA for target embeddings...")
    target_folder = os.path.join(base_path, 'sub-01')  # Load from any subject as target embeddings are the same
    target_embeddings = load_embeddings(target_folder)
    
    target_data = target_embeddings[target_embeddings['embedding_type'] == target_type]
    target_embedding_values = np.array(target_data['embeddings'].tolist())
    
    # Perform PCA on target embeddings with all components
    pca_full = PCA()  # Keep all components to compute the full variance
    pca_full.fit(target_embedding_values)
    cumulative_explained_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100  # Cumulative variance
    
    # Plot cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('Cumulative Explained Variance by PCA Components')
    cumulative_save_path = os.path.join(save_path, target_type, 'cumulative_explained_variance.png')
    os.makedirs(os.path.dirname(cumulative_save_path), exist_ok=True)
    plt.savefig(cumulative_save_path)
    plt.close()
    print(f"Cumulative Explained Variance plot saved to {cumulative_save_path}")
    
    # Reduce to 2 components for the PCA plot
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(target_embedding_values)
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
    
    # Save target PCA results
    target_pca_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Image Path': target_data['original_img_path'],
        'Category': target_data['category'],
        'Split': target_data['split'],
        'Subject': 'Target',
        'Embedding Type': target_type
    })
    
    # Step 2: Process trained embeddings subject by subject
    all_pca_results = [] if not include_target else [target_pca_df]  # Include target PCA results if specified
    if include_all_subjects:
        subjects_to_process = range(1, 11)
    elif subject_id:
        subjects_to_process = [subject_id]
    else:
        subjects_to_process = []  # No subjects, just target
    
    for subj_id in subjects_to_process:
        subject_folder = os.path.join(base_path, f'sub-{subj_id:02d}')
        print(f'Processing trained embeddings for {subject_folder}...')
        
        # Load trained embeddings
        trained_embeddings = load_embeddings(subject_folder)
        trained_data = trained_embeddings[trained_embeddings['embedding_type'] == trained_type]
        trained_embedding_values = np.array(trained_data['embeddings'].tolist())
        
        # Project trained embeddings onto PCA space
        trained_pca_result = pca.transform(trained_embedding_values)
        
        # Save trained PCA results
        trained_pca_df = pd.DataFrame({
            'PCA1': trained_pca_result[:, 0],
            'PCA2': trained_pca_result[:, 1],
            'Image Path': trained_data['original_img_path'],
            'Category': trained_data['category'],
            'Split': trained_data['split'],
            'Subject': f'sub-{subj_id:02d}',
            'Embedding Type': trained_type
        })
        
        all_pca_results.append(trained_pca_df)
    
    # Combine all PCA results
    combined_pca_df = pd.concat(all_pca_results, ignore_index=True)
    
    # Ensure test points are plotted last
    if 'Split' in combined_pca_df.columns:
        combined_pca_df['Split'] = pd.Categorical(combined_pca_df['Split'], categories=['train', 'test'], ordered=True)
    
    # Step 3: Create the PCA plot
    fig = px.scatter(
        combined_pca_df,
        x='PCA1',
        y='PCA2',
        color=color_by,
        symbol='Embedding Type',
        hover_data=['Image Path', 'Subject', 'Split', 'Category'],
        title=f'PCA of Target and Trained Embeddings',
        labels={
            'PCA1': f'PCA1 (EV: {explained_variance[0]:.2f}%)',
            'PCA2': f'PCA2 (EV: {explained_variance[1]:.2f}%)'
        },
        color_discrete_sequence=px.colors.qualitative.Light24  # Updated color scheme
    )
    
    # Update font style and layout
    fig.update_layout(
        font=dict(family="Editorial New Italic", size=16),
        width=width,
        height=height
    )
    
    # Optionally save the plot
    if save_path and filename:
        save_file_path = os.path.join(save_path, target_type, filename)
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        pio.write_image(fig, save_file_path, width=width, height=height, scale=scale)
        print(f"Plot saved to {save_file_path}")

    return fig  # Return the figure

    
def plot_single_subject_pca(base_path, subject_id, embedding_type='ATMS',
                            color_by='Category', save_path='/work3/s184984/repos/EEG_Image_decode/Generation/results/PCA',
                            filename='single_subject_pca.png', width=1000, height=500, scale=None):
    """
    Fit PCA on a single subject's embeddings and visualize the results.
    """
    # Step 1: Load embeddings for the specified subject
    subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
    print(f"Loading embeddings for subject {subject_id} from {subject_folder}...")
    embeddings = load_embeddings(subject_folder)
    
    # Filter by the specified embedding type
    subject_data = embeddings[embeddings['embedding_type'] == embedding_type]
    embedding_values = np.array(subject_data['embeddings'].tolist())
    
    # Step 2: Perform PCA on the embeddings
    print(f"Performing PCA on subject {subject_id} embeddings...")
    pca = PCA(n_components=2)  # Reduce to 2 components for visualization
    pca_result = pca.fit_transform(embedding_values)
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
    
    # Step 3: Create a DataFrame for the PCA results
    pca_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Image Path': subject_data['original_img_path'],
        'Category': subject_data['category'],
        'Split': subject_data['split'],
        'Subject': f'sub-{subject_id:02d}',
        'Embedding Type': embedding_type
    })
    
    # Step 4: Create the PCA plot
    print("Creating PCA plot...")
    fig = px.scatter(
        pca_df,
        x='PCA1',
        y='PCA2',
        color=color_by,
        hover_data=['Image Path', 'Split', 'Category'],
        title=f'PCA of Embeddings for Subject {subject_id}',
        labels={
            'PCA1': f'PCA1 (EV: {explained_variance[0]:.2f}%)',
            'PCA2': f'PCA2 (EV: {explained_variance[1]:.2f}%)'
        }
    )
    
    # Update layout with consistent style
    fig.update_layout(
        font=dict(family="Editorial New Italic", size=16),
        width=width,
        height=height
    )
    
    # Step 5: Save the plot (optional)
    if save_path and filename:
        save_file_path = os.path.join(save_path, f'sub-{subject_id:02d}', filename)
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        pio.write_image(fig, save_file_path, width=width, height=height, scale=scale)
        print(f"Plot saved to {save_file_path}")

    return fig  # Return the figure



def plot_pca_embeddings_global_subset(base_path, target_type='IMG', trained_type='ATMS', 
                                      color_by='Split', subset_fraction=0.1, include_all_subjects=True, 
                                      subject_id=None, include_target=True, 
                                      save_path='/work3/s184984/repos/EEG_Image_decode/Generation/results/PCA', 
                                      filename='pca_plot_global_subset.png', width=1500, height=500, scale=None):
    from sklearn.utils import shuffle
    
    # Step 1: Load and sample subsets for PCA training
    print("Sampling subsets of target and subject embeddings for PCA training...")
    
    # Load target embeddings and sample a subset
    target_folder = os.path.join(base_path, 'sub-01')  # Load from any subject as target embeddings are the same
    target_embeddings = load_embeddings(target_folder)
    target_data = target_embeddings[target_embeddings['embedding_type'] == target_type]
    target_embedding_values = np.array(target_data['embeddings'].tolist())
    target_subset_size = int(len(target_embedding_values) * subset_fraction)
    target_subset = shuffle(target_embedding_values, random_state=42)[:target_subset_size]
    
    # Collect subsets from all subjects
    subject_subsets = []
    if include_all_subjects:
        subjects_to_process = range(1, 11)
    elif subject_id:
        subjects_to_process = [subject_id]
    else:
        subjects_to_process = []
    
    for subj_id in subjects_to_process:
        subject_folder = os.path.join(base_path, f'sub-{subj_id:02d}')
        print(f'Sampling subset of trained embeddings for {subject_folder}...')
        
        # Load trained embeddings and sample a subset
        trained_embeddings = load_embeddings(subject_folder)
        trained_data = trained_embeddings[trained_embeddings['embedding_type'] == trained_type]
        trained_embedding_values = np.array(trained_data['embeddings'].tolist())
        subset_size = int(len(trained_embedding_values) * subset_fraction)
        trained_subset = shuffle(trained_embedding_values, random_state=42)[:subset_size]
        subject_subsets.append(trained_subset)
    
    # Combine all subsets for PCA training
    combined_subset = np.vstack([target_subset] + subject_subsets)
    
    # Step 2: Train PCA on the combined subset
    print("Training PCA on the combined subset...")
    pca = PCA(n_components=2)
    pca.fit(combined_subset)
    
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
    
    # Project target embeddings into PCA space
    target_pca_result = pca.transform(target_embedding_values)
    target_pca_df = pd.DataFrame({
        'PCA1': target_pca_result[:, 0],
        'PCA2': target_pca_result[:, 1],
        'Image Path': target_data['original_img_path'],
        'Category': target_data['category'],
        'Split': target_data['split'],
        'Subject': 'Target',
        'Embedding Type': 'Target Projected'
    })
    
    # Step 3: Project all subject embeddings into PCA space
    all_pca_results = [] if not include_target else [target_pca_df]
    
    for subj_id in subjects_to_process:
        subject_folder = os.path.join(base_path, f'sub-{subj_id:02d}')
        print(f'Processing trained embeddings for {subject_folder}...')
        
        trained_embeddings = load_embeddings(subject_folder)
        trained_data = trained_embeddings[trained_embeddings['embedding_type'] == trained_type]
        trained_embedding_values = np.array(trained_data['embeddings'].tolist())
        
        # Project embeddings into PCA space
        trained_pca_result = pca.transform(trained_embedding_values)
        trained_pca_df = pd.DataFrame({
            'PCA1': trained_pca_result[:, 0],
            'PCA2': trained_pca_result[:, 1],
            'Image Path': trained_data['original_img_path'],
            'Category': trained_data['category'],
            'Split': trained_data['split'],
            'Subject': f'sub-{subj_id:02d}',
            'Embedding Type': 'Trained Projected'
        })
        
        all_pca_results.append(trained_pca_df)
    
    # Combine all PCA results
    combined_pca_df = pd.concat(all_pca_results, ignore_index=True)
    
    # Ensure test points are plotted last
    if 'Split' in combined_pca_df.columns:
        combined_pca_df['Split'] = pd.Categorical(combined_pca_df['Split'], categories=['train', 'test'], ordered=True)
    
    # Step 4: Create the PCA plot
    fig = px.scatter(
        combined_pca_df,
        x='PCA1',
        y='PCA2',
        color=color_by,
        symbol='Embedding Type',
        hover_data=['Image Path', 'Subject', 'Split', 'Category'],
        title=f'PCA of Subset-Trained Target and Trained Embeddings',
        labels={
            'PCA1': f'PCA1 (EV: {explained_variance[0]:.2f}%)',
            'PCA2': f'PCA2 (EV: {explained_variance[1]:.2f}%)'
        },
        color_discrete_map={'train': 'blue', 'test': 'red', 'Target': 'green'}  # Fixed colors
    )
    
    # Update font style and layout
    fig.update_layout(
        font=dict(family="Editorial New Italic", size=16),
        width=width,
        height=height
    )
    
    # Optionally save the plot
    if save_path and filename:
        save_file_path = os.path.join(save_path, target_type, filename)
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        pio.write_image(fig, save_file_path, width=width, height=height, scale=scale)
        print(f"Plot saved to {save_file_path}")

    return fig  # Return the figure


def plot_pca_random_split(base_path, target_type='IMG', subset_fraction=0.5, 
                          save_path='/work3/s184984/repos/EEG_Image_decode/Generation/results/PCA', 
                          filename='pca_plot_random_split.png', width=1500, height=500, scale=None):
    # Step 1: Load target embeddings
    print("Loading target embeddings...")
    target_folder = os.path.join(base_path, 'sub-01')  # Load from any subject as target embeddings are the same
    target_embeddings = load_embeddings(target_folder)
    
    target_data = target_embeddings[target_embeddings['embedding_type'] == target_type]
    target_embedding_values = np.array(target_data['embeddings'].tolist())
    
    # Step 2: Randomly split into training and projection subsets
    print("Splitting target embeddings into training and projection subsets...")
    num_samples = len(target_embedding_values)
    subset_size = int(subset_fraction * num_samples)
    random_indices = np.random.permutation(num_samples)
    
    training_indices = random_indices[:subset_size]
    projection_indices = random_indices[subset_size:]
    
    training_embeddings = target_embedding_values[training_indices]
    projection_embeddings = target_embedding_values[projection_indices]
    
    # Step 3: Perform PCA on the training subset
    print("Performing PCA on the training subset...")
    pca = PCA(n_components=2)
    pca_training_result = pca.fit_transform(training_embeddings)
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
    
    # Step 4: Project the remaining embeddings onto the same PCA space
    print("Projecting remaining embeddings onto the PCA space...")
    pca_projection_result = pca.transform(projection_embeddings)
    
    # Step 5: Prepare DataFrames for visualization
    training_pca_df = pd.DataFrame({
        'PCA1': pca_training_result[:, 0],
        'PCA2': pca_training_result[:, 1],
        'Image Path': target_data.iloc[training_indices]['original_img_path'].values,
        'Category': target_data.iloc[training_indices]['category'].values,
        'Split': 'Training',
        'Embedding Type': target_type
    })
    
    projection_pca_df = pd.DataFrame({
        'PCA1': pca_projection_result[:, 0],
        'PCA2': pca_projection_result[:, 1],
        'Image Path': target_data.iloc[projection_indices]['original_img_path'].values,
        'Category': target_data.iloc[projection_indices]['category'].values,
        'Split': 'Projection',
        'Embedding Type': target_type
    })
    
    combined_pca_df = pd.concat([training_pca_df, projection_pca_df], ignore_index=True)
    
    # Step 6: Create the PCA plot
    print("Creating PCA plot...")
    fig = px.scatter(
        combined_pca_df,
        x='PCA1',
        y='PCA2',
        color='Split',
        hover_data=['Image Path', 'Category'],
        title=f'PCA of Randomly Split Target Embeddings',
        labels={
            'PCA1': f'PCA1 (EV: {explained_variance[0]:.2f}%)',
            'PCA2': f'PCA2 (EV: {explained_variance[1]:.2f}%)'
        },
        color_discrete_map={'Training': 'green', 'Projection': 'lightgreen'}
    )
    
    # Update layout
    fig.update_layout(
        font=dict(family="Editorial New Italic", size=16),
        width=width,
        height=height
    )
    
    # Optionally save the plot
    if save_path and filename:
        save_file_path = os.path.join(save_path, target_type, filename)
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        pio.write_image(fig, save_file_path, width=width, height=height, scale=scale)
        print(f"Plot saved to {save_file_path}")

    return fig  # Return the figure


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def combine_subject_plots(base_path, color_by, filename, target_type='IMG', trained_type='ATMS', nrows=2, ncols=5, vertical_spacing=0.07, legend_font_size=14):
    # Create a subplot layout with reduced vertical spacing
    fig = make_subplots(
        rows=nrows, 
        cols=ncols, 
        subplot_titles=[f"Subject {i:02d}" for i in range(1, 11)],
        vertical_spacing=vertical_spacing
    )
    
    legend_shown = set()  # Track which legend entries have been added
    
    for subject_id in range(1, 11):
        # Generate PCA plot for the subject
        temp_fig = plot_pca_embeddings(
            base_path=base_path,
            target_type=target_type, 
            trained_type=trained_type,
            include_all_subjects=False,
            subject_id=subject_id,
            include_target=False,
            color_by=color_by,
            filename=None  # Don't save individual plots
        )
        
        # Add each trace to the combined subplot
        for trace in temp_fig.data:
            # Check if this legend entry has already been added
            show_legend = trace.name not in legend_shown
            if show_legend:
                legend_shown.add(trace.name)
            
            # Extract relevant marker attributes explicitly
            marker_props = {
                'color': trace.marker.color,
                'size': trace.marker.size,
                'symbol': trace.marker.symbol,
                'opacity': trace.marker.opacity
            }
            
            # Add the trace to the correct subplot
            row = (subject_id - 1) // ncols + 1
            col = (subject_id - 1) % ncols + 1
            fig.add_trace(
                go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    marker=marker_props,
                    name=trace.name,
                    legendgroup=trace.legendgroup,  # Ensure consistent legend grouping
                    showlegend=show_legend  # Only show legend for the first occurrence
                ),
                row=row,
                col=col
            )
    
    # Update layout for the combined plot
    fig.update_layout(
        # title_text=f"Combined PCA Plots Colored by {color_by}",
        height=1000,  # Adjust as needed
        width=1500,
        font=dict(family="Editorial New Italic", size=20),  # Set custom font
        legend=dict(
            title=f"{color_by}",
            font=dict(size=legend_font_size),  # Adjust legend font size
            x=1.02,  # Position the legend to the right of the plot
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        annotations=[dict(font=dict(size=24)) for a in fig['layout']['annotations']],
        showlegend=True  # Enable a single legend for the entire plot
    )
    
    # Save the combined plot
    pio.write_image(fig, f"/work3/s184984/repos/EEG_Image_decode/Generation/results/PCA/{target_type}/{filename}", scale=2)
    print(f"Combined plot saved as {filename}")
    
    

def combine_subject_plots_EEG_space(base_path, color_by, filename, target_type='IMG', trained_type='ATMS', nrows=2, ncols=5, vertical_spacing=0.07, legend_font_size=14):
    # Create a subplot layout with reduced vertical spacing
    fig = make_subplots(
        rows=nrows, 
        cols=ncols, 
        subplot_titles=[f"Subject {i:02d}" for i in range(1, 11)],
        vertical_spacing=vertical_spacing
    )
    
    legend_shown = set()  # Track which legend entries have been added
    
    for subject_id in range(1, 11):
        # Generate PCA plot for the subject
        
        temp_fig = plot_single_subject_pca(
            base_path=base_path, 
            subject_id=subject_id, 
            embedding_type=trained_type,
            color_by=color_by, 
            filename=None
        )     
        
        
        # Add each trace to the combined subplot
        for trace in temp_fig.data:
            # Check if this legend entry has already been added
            show_legend = trace.name not in legend_shown
            if show_legend:
                legend_shown.add(trace.name)
            
            # Extract relevant marker attributes explicitly
            marker_props = {
                'color': trace.marker.color,
                'size': trace.marker.size,
                'symbol': trace.marker.symbol,
                'opacity': trace.marker.opacity
            }
            
            # Add the trace to the correct subplot
            row = (subject_id - 1) // ncols + 1
            col = (subject_id - 1) % ncols + 1
            fig.add_trace(
                go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    marker=marker_props,
                    name=trace.name,
                    legendgroup=trace.legendgroup,  # Ensure consistent legend grouping
                    showlegend=show_legend  # Only show legend for the first occurrence
                ),
                row=row,
                col=col
            )
    
    # Update layout for the combined plot
    fig.update_layout(
        # title_text=f"Combined PCA Plots Colored by {color_by}",
        height=1000,  # Adjust as needed
        width=1500,
        font=dict(family="Editorial New Italic", size=20),  # Set custom font
        legend=dict(
            title=f"{color_by}",
            font=dict(size=legend_font_size),  # Adjust legend font size
            x=1.02,  # Position the legend to the right of the plot
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        annotations=[dict(font=dict(size=24)) for a in fig['layout']['annotations']],
        showlegend=True  # Enable a single legend for the entire plot
    )
    
    # Save the combined plot
    pio.write_image(fig, f"/work3/s184984/repos/EEG_Image_decode/Generation/results/PCA/{target_type}/{filename}", scale=2)
    print(f"Combined plot saved as {filename}")


import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def heatmap_CS(base_path, subject_id, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    # Step 1: Load and filter embeddings
    subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
    embeddings_df = load_embeddings(subject_folder)
    test_embeddings = embeddings_df[embeddings_df['split'] == 'test']
    
    # Step 2: Align target and trained embeddings by category and file path
    target_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == target_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    trained_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == trained_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    
    # Verify alignment of file paths
    assert list(target_embeddings['original_img_path']) == list(trained_embeddings['original_img_path']), \
        "File paths between target and trained embeddings do not align. Check sorting or filtering."
    
    # Step 3: Compute cosine similarity without normalization
    target_matrix = np.stack(target_embeddings['embeddings'].values)
    trained_matrix = np.stack(trained_embeddings['embeddings'].values)
    confusion_matrix = cosine_similarity(trained_matrix, target_matrix)
    
    # Step 4: Get unique categories and their indices
    unique_categories, category_indices = np.unique(target_embeddings['category'], return_index=True)
    
    # Step 5: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            colorscale='Blues',
            colorbar=dict(title='Cosine Similarity'),
            xgap=0.05,  # Small gaps for subtle separation
            ygap=0.05
        )
    )
    
    # Step 6: Add category borders and labels
    for i, (category, start_idx) in enumerate(zip(unique_categories, category_indices)):
        end_idx = category_indices[i + 1] - 1 if i + 1 < len(category_indices) else len(target_embeddings) - 1
        
        # Add x-axis category bar
        fig.add_shape(
            type="rect",
            x0=start_idx - 0.5, x1=end_idx + 0.5,
            y0=-bar_size, y1=-0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=(start_idx + end_idx) / 2,
            y=-bar_size / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=90
        )
        
        # Add y-axis category bar
        fig.add_shape(
            type="rect",
            x0=-bar_size, x1=-0.5,
            y0=start_idx - 0.5, y1=end_idx + 0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=-bar_size / 2,
            y=(start_idx + end_idx) / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=0
        )
    
    # Step 7: Update layout for square plot and white background
    fig.update_layout(
        title=f'Cosine Heatmap - Subject {subject_id}',
        font=dict(family="Editorial New Italic", size=16),
        width=600 + bar_size,
        height=600 + bar_size,
        xaxis=dict(title='CLIP Target Embeddings', tickvals=[], ticktext=[], scaleanchor='y'),
        yaxis=dict(title='EEG Trained Embeddings', tickvals=[], ticktext=[]),
        plot_bgcolor='white',  # Set background to white
        paper_bgcolor='white'  # Remove greyish paper background
    )
    
    # Step 8: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Heatmaps"
    os.makedirs(save_path, exist_ok=True)
    filename = f'subject_{subject_id:02d}_{trained_type}_vs_{target_type}_CS.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Heatmap saved: {os.path.join(save_path, filename)}")
    
def heatmap_CS_all_subjects(base_path, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    import os
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import plotly.graph_objects as go

    # Step 1: Initialize variables for averaging
    accumulated_confusion_matrix = None
    num_subjects = 10
    aggregated_categories = None

    # Step 2: Iterate over all subjects and compute cosine similarity matrices
    for subject_id in range(1, num_subjects + 1):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        test_embeddings = embeddings_df[embeddings_df['split'] == 'test']

        # Align target and trained embeddings by category and file path
        target_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == target_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )
        trained_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == trained_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )

        # Verify alignment of file paths
        assert list(target_embeddings['original_img_path']) == list(trained_embeddings['original_img_path']), \
            f"File paths between target and trained embeddings do not align for subject {subject_id}. Check sorting or filtering."

        # Compute cosine similarity without normalization
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        confusion_matrix = cosine_similarity(trained_matrix, target_matrix)

        # Initialize accumulated matrix or add to it
        if accumulated_confusion_matrix is None:
            accumulated_confusion_matrix = confusion_matrix
            aggregated_categories = target_embeddings['category'].values
        else:
            accumulated_confusion_matrix += confusion_matrix

    # Step 3: Average the confusion matrix
    averaged_confusion_matrix = accumulated_confusion_matrix / num_subjects

    # Step 4: Get unique categories and their indices
    unique_categories, category_indices = np.unique(aggregated_categories, return_index=True)

    # Step 5: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=averaged_confusion_matrix,
            colorscale='Blues',
            colorbar=dict(title='Average Cosine Similarity'),
            xgap=0.05,  # Small gaps for subtle separation
            ygap=0.05
        )
    )

    # Step 6: Add category borders and labels
    for i, (category, start_idx) in enumerate(zip(unique_categories, category_indices)):
        end_idx = category_indices[i + 1] - 1 if i + 1 < len(category_indices) else len(aggregated_categories) - 1

        # Add x-axis category bar
        fig.add_shape(
            type="rect",
            x0=start_idx - 0.5, x1=end_idx + 0.5,
            y0=-bar_size, y1=-0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=(start_idx + end_idx) / 2,
            y=-bar_size / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=90
        )

        # Add y-axis category bar
        fig.add_shape(
            type="rect",
            x0=-bar_size, x1=-0.5,
            y0=start_idx - 0.5, y1=end_idx + 0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=-bar_size / 2,
            y=(start_idx + end_idx) / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=0
        )

    # Step 7: Update layout for square plot and white background
    fig.update_layout(
        title=f'Average Cosine Heatmap (All Subjects)',
        font=dict(family="Editorial New Italic", size=16),
        width=600 + bar_size,
        height=600 + bar_size,
        xaxis=dict(title='CLIP Target Embeddings', tickvals=[], ticktext=[], scaleanchor='y'),
        yaxis=dict(title='EEG Trained Embeddings', tickvals=[], ticktext=[]),
        plot_bgcolor='white',  # Set background to white
        paper_bgcolor='white'  # Remove greyish paper background
    )

    # Step 8: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Heatmaps"
    os.makedirs(save_path, exist_ok=True)
    filename = f'average_{trained_type}_vs_{target_type}_CS.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Average heatmap saved: {os.path.join(save_path, filename)}")



import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def heatmap_MSE(base_path, subject_id, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    # Step 1: Load and filter embeddings
    subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
    embeddings_df = load_embeddings(subject_folder)
    test_embeddings = embeddings_df[embeddings_df['split'] == 'test']
    
    # Step 2: Align target and trained embeddings by category and file path (alphabetically)
    target_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == target_type]
        .sort_values(by=['category', 'original_img_path'], ascending=True)
        .head(200)
    )
    trained_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == trained_type]
        .sort_values(by=['category', 'original_img_path'], ascending=True)
        .head(200)
    )
    
    # Verify alignment of file paths
    assert list(target_embeddings['original_img_path']) == list(trained_embeddings['original_img_path']), \
        "File paths between target and trained embeddings do not align. Check sorting or filtering."
    
    # Step 3: Compute MSE
    target_matrix = np.stack(target_embeddings['embeddings'].values)
    trained_matrix = np.stack(trained_embeddings['embeddings'].values)
    mse_matrix = np.mean((trained_matrix[:, np.newaxis] - target_matrix) ** 2, axis=2)
    
    # Step 4: Get unique categories and their indices (alphabetically sorted)
    unique_categories, category_indices = np.unique(target_embeddings['category'], return_index=True)
    
    # Step 5: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=mse_matrix,
            colorscale='Reds',
            colorbar=dict(title='Mean Squared Error'),
            xgap=0.05,
            ygap=0.05
        )
    )
    
    # Step 6: Add properly aligned category borders and labels
    for i, (category, start_idx) in enumerate(zip(unique_categories, category_indices)):
        end_idx = category_indices[i + 1] - 1 if i + 1 < len(category_indices) else len(target_embeddings) - 1
        
        # Add x-axis category bar (align borders exactly with the indices)
        fig.add_shape(
            type="rect",
            x0=start_idx - 0.5, x1=end_idx + 0.5,
            y0=-bar_size, y1=-0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=(start_idx + end_idx) / 2,
            y=-bar_size / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=90
        )
        
        # Add y-axis category bar (align borders exactly with the indices)
        fig.add_shape(
            type="rect",
            x0=-bar_size, x1=-0.5,
            y0=start_idx - 0.5, y1=end_idx + 0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=-bar_size / 2,
            y=(start_idx + end_idx) / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=0
        )
    
    # Step 7: Update layout for square plot and white background
    fig.update_layout(
        title=f'MSE Heatmap - Subject {subject_id}',
        font=dict(family="Editorial New Italic", size=16),
        width=600 + bar_size,
        height=600 + bar_size,
        xaxis=dict(title='CLIP Target Embeddings', tickvals=[], ticktext=[], scaleanchor='y'),
        yaxis=dict(title='EEG Categories', tickvals=[], ticktext=[]),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Step 8: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Heatmaps"
    os.makedirs(save_path, exist_ok=True)
    filename = f'subject_{subject_id:02d}_{trained_type}_vs_{target_type}_MSE.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Heatmap saved: {os.path.join(save_path, filename)}")
    
def heatmap_MSE_all_subjects(base_path, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    import os
    import numpy as np
    import plotly.graph_objects as go

    # Step 1: Initialize variables for averaging
    accumulated_mse_matrix = None
    num_subjects = 10
    aggregated_categories = None

    # Step 2: Iterate over all subjects and compute MSE matrices
    for subject_id in range(1, num_subjects + 1):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        test_embeddings = embeddings_df[embeddings_df['split'] == 'test']

        # Align target and trained embeddings by category and file path (alphabetically)
        target_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == target_type]
            .sort_values(by=['category', 'original_img_path'], ascending=True)
            .head(200)
        )
        trained_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == trained_type]
            .sort_values(by=['category', 'original_img_path'], ascending=True)
            .head(200)
        )

        # Verify alignment of file paths
        assert list(target_embeddings['original_img_path']) == list(trained_embeddings['original_img_path']), \
            f"File paths between target and trained embeddings do not align for subject {subject_id}. Check sorting or filtering."

        # Compute MSE
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        mse_matrix = np.mean((trained_matrix[:, np.newaxis] - target_matrix) ** 2, axis=2)

        # Initialize accumulated matrix or add to it
        if accumulated_mse_matrix is None:
            accumulated_mse_matrix = mse_matrix
            aggregated_categories = target_embeddings['category'].values
        else:
            accumulated_mse_matrix += mse_matrix

    # Step 3: Average the MSE matrix
    averaged_mse_matrix = accumulated_mse_matrix / num_subjects

    # Step 4: Get unique categories and their indices (alphabetically sorted)
    unique_categories, category_indices = np.unique(aggregated_categories, return_index=True)

    # Step 5: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=averaged_mse_matrix,
            colorscale='Reds',
            colorbar=dict(title='Average Mean Squared Error'),
            xgap=0.05,
            ygap=0.05
        )
    )

    # Step 6: Add properly aligned category borders and labels
    for i, (category, start_idx) in enumerate(zip(unique_categories, category_indices)):
        end_idx = category_indices[i + 1] - 1 if i + 1 < len(category_indices) else len(aggregated_categories) - 1

        # Add x-axis category bar (align borders exactly with the indices)
        fig.add_shape(
            type="rect",
            x0=start_idx - 0.5, x1=end_idx + 0.5,
            y0=-bar_size, y1=-0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=(start_idx + end_idx) / 2,
            y=-bar_size / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=90
        )

        # Add y-axis category bar (align borders exactly with the indices)
        fig.add_shape(
            type="rect",
            x0=-bar_size, x1=-0.5,
            y0=start_idx - 0.5, y1=end_idx + 0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=-bar_size / 2,
            y=(start_idx + end_idx) / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=0
        )

    # Step 7: Update layout for square plot and white background
    fig.update_layout(
        title=f'Average MSE Heatmap (All Subjects)',
        font=dict(family="Editorial New Italic", size=16),
        width=600 + bar_size,
        height=600 + bar_size,
        xaxis=dict(title='CLIP Target Embeddings', tickvals=[], ticktext=[], scaleanchor='y'),
        yaxis=dict(title='EEG Categories', tickvals=[], ticktext=[]),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Step 8: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Heatmaps"
    os.makedirs(save_path, exist_ok=True)
    filename = f'average_{trained_type}_vs_{target_type}_MSE.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Average heatmap saved: {os.path.join(save_path, filename)}")



import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics.pairwise import cosine_similarity
from plotly.express.colors import qualitative
import random
import re

def topk_kway(base_path, target_type='IMG', trained_type='ATMS', save_path=None, width=1500, height=400, scale=2, seed=42):
    """
    Compute accuracy, top-5 accuracy, and K-way scores for all subjects and plot bar plots.
    The K-way scores include 2-way, 4-way, and 10-way accuracy with random sampling.
    """
    random.seed(seed)

    def load_subject_embeddings(subject_id):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        return embeddings_df[embeddings_df['split'] == 'test']

    def calculate_metrics(subject_id):
        embeddings_df = load_subject_embeddings(subject_id)
        
        target_embeddings = (
            embeddings_df[embeddings_df['embedding_type'] == target_type]
            .sort_values(by='original_img_path')
            .head(200)
        )
        trained_embeddings = (
            embeddings_df[embeddings_df['embedding_type'] == trained_type]
            .sort_values(by='original_img_path')
            .head(200)
        )
        
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        similarity_matrix = cosine_similarity(trained_matrix, target_matrix)
        
        correct_filepaths = list(target_embeddings['original_img_path'])
        
        accuracy = 0
        top5_accuracy = 0
        kway_scores = {2: 0, 4: 0, 10: 0}
        
        for i, row in enumerate(similarity_matrix):
            sorted_indices = np.argsort(-row)
            sorted_filepaths = [correct_filepaths[idx] for idx in sorted_indices]
            
            if correct_filepaths[i] == sorted_filepaths[0]:
                accuracy += 1
            if correct_filepaths[i] in sorted_filepaths[:5]:
                top5_accuracy += 1

            for k in kway_scores.keys():
                sampled_indices = random.sample([idx for idx in range(len(correct_filepaths)) if idx != i], k - 1)
                sampled_filepaths = [correct_filepaths[idx] for idx in sampled_indices] + [correct_filepaths[i]]
                sampled_similarities = [row[idx] for idx in sampled_indices] + [row[i]]
                
                if np.argmax(sampled_similarities) == len(sampled_similarities) - 1:
                    kway_scores[k] += 1
        
        total = len(correct_filepaths)
        return {
            'accuracy': accuracy / total,
            'top5_accuracy': top5_accuracy / total,
            **{f'{k}-way': kway_scores[k] / total for k in kway_scores}
        }

    subjects = [f'sub-{i:02d}' for i in range(1, 11)]
    subject_scores = {}
    for subject_id in range(1, 11):
        subject_scores[f'sub-{subject_id:02d}'] = calculate_metrics(subject_id)

    metrics = ['accuracy', 'top5_accuracy', '2-way', '4-way', '10-way']
    colors = qualitative.Plotly
    metric_stats = {metric: [] for metric in metrics}

    for metric in metrics:
        scores = [subject_scores[subject][metric] for subject in subjects]
        metric_stats[metric] = {
            'mean': np.mean(scores),
            'stddev': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'min_subject': subjects[np.argmin(scores)],
            'max_subject': subjects[np.argmax(scores)]
        }

    fig = go.Figure()

    for i, metric in enumerate(metrics):
        stats = metric_stats[metric]

        fig.add_trace(
            go.Bar(
                x=[metric],
                y=[stats['mean']],
                name=f"{metric.capitalize()} Mean",
                error_y=dict(type='data', array=[stats['stddev']]),
                marker_color=colors[0],
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[metric],
                y=[stats['min']],
                mode='markers',
                marker=dict(color=colors[subjects.index(stats['min_subject']) % len(colors)], size=10),
                name=stats['min_subject'],
                legendgroup=stats['min_subject'],
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[metric],
                y=[stats['max']],
                mode='markers',
                marker=dict(color=colors[subjects.index(stats['max_subject']) % len(colors)], size=10),
                name=stats['max_subject'],
                legendgroup=stats['max_subject'],
                showlegend=False
            )
        )

    for subject in subjects:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=colors[subjects.index(subject) % len(colors)], size=10),
                name=subject,
                legendgroup=subject,
                showlegend=True
            )
        )

    fig.update_layout(
        # title="Accuracies and K-way Scores",
        xaxis=dict(title="Metrics", tickfont=dict(size=18)),
        yaxis=dict(title="Score", range=[0, 1.05], tickfont=dict(size=18)),
        legend=dict(title="Subjects", orientation="h", y=-.3, x=0),
        font=dict(family="Editorial New Italic", size=16),
        barmode='group',
        margin=dict(b=200)
    )

    if save_path:
        pio.write_image(fig, save_path, width=width, height=height, scale=scale)

    else:
        fig.show()


def confusion_matrix_CS(base_path, subject_id, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    # Step 1: Load and filter embeddings
    subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
    embeddings_df = load_embeddings(subject_folder)
    test_embeddings = embeddings_df[embeddings_df['split'] == 'test']
    
    # Step 2: Align target and trained embeddings by category and file path
    target_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == target_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    trained_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == trained_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    
    # Verify alignment of file paths
    assert list(target_embeddings['original_img_path']) == list(trained_embeddings['original_img_path']), \
        "File paths between target and trained embeddings do not align. Check sorting or filtering."
    
    # Step 3: Compute cosine similarity
    target_matrix = np.stack(target_embeddings['embeddings'].values)
    trained_matrix = np.stack(trained_embeddings['embeddings'].values)
    similarity_matrix = cosine_similarity(trained_matrix, target_matrix)
    
    # Step 4: Generate confusion matrix
    # For each trained embedding, find the index of the most similar target embedding
    predicted_indices = np.argmax(similarity_matrix, axis=1)
    confusion_matrix = np.zeros((200, 200))
    
    for i, predicted_idx in enumerate(predicted_indices):
        confusion_matrix[i, predicted_idx] = 1  # Mark the predicted target embedding for each trained embedding
    
    # Step 5: Get unique categories and their indices
    unique_categories, category_indices = np.unique(target_embeddings['category'], return_index=True)
    
    # Step 6: Create the confusion matrix heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            colorscale='Blues',
            colorbar=dict(title='Classification'),
            xgap=0.05,  # Small gaps for subtle separation
            ygap=0.05
        )
    )
    
    # Step 7: Add category borders and labels
    for i, (category, start_idx) in enumerate(zip(unique_categories, category_indices)):
        end_idx = category_indices[i + 1] - 1 if i + 1 < len(category_indices) else len(target_embeddings) - 1
        
        # Add x-axis category bar
        fig.add_shape(
            type="rect",
            x0=start_idx - 0.5, x1=end_idx + 0.5,
            y0=-bar_size, y1=-0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=(start_idx + end_idx) / 2,
            y=-bar_size / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=90
        )
        
        # Add y-axis category bar
        fig.add_shape(
            type="rect",
            x0=-bar_size, x1=-0.5,
            y0=start_idx - 0.5, y1=end_idx + 0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=-bar_size / 2,
            y=(start_idx + end_idx) / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=0
        )
    
    # Step 8: Update layout for square plot and white background
    fig.update_layout(
        title=f'Subject {subject_id}',
        font=dict(family="Editorial New Italic", size=16),
        width=600 + bar_size,
        height=600 + bar_size,
        xaxis=dict(title='Predicted Target Embeddings', tickvals=[], ticktext=[], scaleanchor='y'),
        yaxis=dict(title='True Trained Embeddings', tickvals=[], ticktext=[]),
        plot_bgcolor='white',  # Set background to white
        paper_bgcolor='white'  # Remove greyish paper background
    )
    
    # Step 9: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Confusion_Matrices"
    os.makedirs(save_path, exist_ok=True)
    filename = f'subject_{subject_id:02d}_{trained_type}_vs_{target_type}_confusion_matrix.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Confusion matrix saved: {os.path.join(save_path, filename)}")


def confusion_matrix_CS_all_subjects(base_path, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    import os
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import plotly.graph_objects as go

    # Step 1: Initialize variables for averaging
    accumulated_confusion_matrix = np.zeros((200, 200))
    num_subjects = 10
    aggregated_categories = None

    # Step 2: Iterate over all subjects and compute confusion matrices
    for subject_id in range(1, num_subjects + 1):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        test_embeddings = embeddings_df[embeddings_df['split'] == 'test']

        # Align target and trained embeddings by category and file path
        target_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == target_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )
        trained_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == trained_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )

        # Verify alignment of file paths
        assert list(target_embeddings['original_img_path']) == list(trained_embeddings['original_img_path']), \
            f"File paths between target and trained embeddings do not align for subject {subject_id}. Check sorting or filtering."

        # Compute cosine similarity
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        similarity_matrix = cosine_similarity(trained_matrix, target_matrix)

        # Generate confusion matrix
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        confusion_matrix = np.zeros((200, 200))

        for i, predicted_idx in enumerate(predicted_indices):
            confusion_matrix[i, predicted_idx] = 1

        # Accumulate confusion matrices
        accumulated_confusion_matrix += confusion_matrix

    # Step 3: Average the confusion matrix
    averaged_confusion_matrix = accumulated_confusion_matrix / num_subjects

    # Step 4: Get unique categories and their indices
    aggregated_categories = target_embeddings['category'].values
    unique_categories, category_indices = np.unique(aggregated_categories, return_index=True)

    # Step 5: Create the confusion matrix heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=averaged_confusion_matrix,
            colorscale='Blues',
            colorbar=dict(title='Average Classification'),
            xgap=0.05,
            ygap=0.05
        )
    )

    # Step 6: Add category borders and labels
    for i, (category, start_idx) in enumerate(zip(unique_categories, category_indices)):
        end_idx = category_indices[i + 1] - 1 if i + 1 < len(category_indices) else len(aggregated_categories) - 1

        # Add x-axis category bar
        fig.add_shape(
            type="rect",
            x0=start_idx - 0.5, x1=end_idx + 0.5,
            y0=-bar_size, y1=-0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=(start_idx + end_idx) / 2,
            y=-bar_size / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=90
        )

        # Add y-axis category bar
        fig.add_shape(
            type="rect",
            x0=-bar_size, x1=-0.5,
            y0=start_idx - 0.5, y1=end_idx + 0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=-bar_size / 2,
            y=(start_idx + end_idx) / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=0
        )

    # Step 7: Update layout for square plot and white background
    fig.update_layout(
        title=f'Average Confusion Matrix (All Subjects)',
        font=dict(family="Editorial New Italic", size=16),
        width=600 + bar_size,
        height=600 + bar_size,
        xaxis=dict(title='Predicted Label', tickvals=[], ticktext=[], scaleanchor='y'),
        yaxis=dict(title='True Label', tickvals=[], ticktext=[]),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Step 8: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Confusion_Matrices"
    os.makedirs(save_path, exist_ok=True)
    filename = f'average_{trained_type}_vs_{target_type}_confusion_matrix.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Average confusion matrix saved: {os.path.join(save_path, filename)}")


def confusion_matrix_MSE(base_path, subject_id, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    # Step 1: Load and filter embeddings
    subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
    embeddings_df = load_embeddings(subject_folder)
    test_embeddings = embeddings_df[embeddings_df['split'] == 'test']
    
    # Step 2: Align target and trained embeddings by category and file path
    target_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == target_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    trained_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == trained_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    
    # Verify alignment of file paths
    assert list(target_embeddings['original_img_path']) == list(trained_embeddings['original_img_path']), \
        "File paths between target and trained embeddings do not align. Check sorting or filtering."
    
    # Step 3: Compute MSE
    target_matrix = np.stack(target_embeddings['embeddings'].values)
    trained_matrix = np.stack(trained_embeddings['embeddings'].values)
    mse_matrix = np.mean((trained_matrix[:, np.newaxis] - target_matrix) ** 2, axis=2)
    
    # Step 4: Generate confusion matrix
    # For each trained embedding, find the index of the target embedding with the lowest MSE
    predicted_indices = np.argmin(mse_matrix, axis=1)
    confusion_matrix = np.zeros((200, 200))
    
    for i, predicted_idx in enumerate(predicted_indices):
        confusion_matrix[i, predicted_idx] = 1  # Mark the predicted target embedding for each trained embedding
    
    # Step 5: Get unique categories and their indices
    unique_categories, category_indices = np.unique(target_embeddings['category'], return_index=True)
    
    # Step 6: Create the confusion matrix heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            colorscale='Reds',  # Use blues as the theme
            colorbar=dict(title='Classification'),
            xgap=0.05,
            ygap=0.05
        )
    )
    
    # Step 7: Add category borders and labels
    for i, (category, start_idx) in enumerate(zip(unique_categories, category_indices)):
        end_idx = category_indices[i + 1] - 1 if i + 1 < len(category_indices) else len(target_embeddings) - 1
        
        # Add x-axis category bar
        fig.add_shape(
            type="rect",
            x0=start_idx - 0.5, x1=end_idx + 0.5,
            y0=-bar_size, y1=-0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=(start_idx + end_idx) / 2,
            y=-bar_size / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=90
        )
        
        # Add y-axis category bar
        fig.add_shape(
            type="rect",
            x0=-bar_size, x1=-0.5,
            y0=start_idx - 0.5, y1=end_idx + 0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=-bar_size / 2,
            y=(start_idx + end_idx) / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=0
        )
    
    # Step 8: Update layout for square plot and white background
    fig.update_layout(
        title=f'Subject {subject_id}',
        font=dict(family="Editorial New Italic", size=16),
        width=600 + bar_size,
        height=600 + bar_size,
        xaxis=dict(title='Predicted Target Embeddings', tickvals=[], ticktext=[], scaleanchor='y'),
        yaxis=dict(title='True Trained Embeddings', tickvals=[], ticktext=[]),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Step 9: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Confusion_Matrices"
    os.makedirs(save_path, exist_ok=True)
    filename = f'subject_{subject_id:02d}_{trained_type}_vs_{target_type}_mse_confusion_matrix.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Confusion matrix saved: {os.path.join(save_path, filename)}")

def confusion_matrix_MSE_all_subjects(base_path, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    import os
    import numpy as np
    import plotly.graph_objects as go

    # Step 1: Initialize variables for averaging
    accumulated_confusion_matrix = np.zeros((200, 200))
    num_subjects = 10
    aggregated_categories = None

    # Step 2: Iterate over all subjects and compute confusion matrices
    for subject_id in range(1, num_subjects + 1):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        test_embeddings = embeddings_df[embeddings_df['split'] == 'test']

        # Align target and trained embeddings by category and file path
        target_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == target_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )
        trained_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == trained_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )

        # Verify alignment of file paths
        assert list(target_embeddings['original_img_path']) == list(trained_embeddings['original_img_path']), \
            f"File paths between target and trained embeddings do not align for subject {subject_id}. Check sorting or filtering."

        # Compute MSE
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        mse_matrix = np.mean((trained_matrix[:, np.newaxis] - target_matrix) ** 2, axis=2)

        # Generate confusion matrix
        predicted_indices = np.argmin(mse_matrix, axis=1)
        confusion_matrix = np.zeros((200, 200))

        for i, predicted_idx in enumerate(predicted_indices):
            confusion_matrix[i, predicted_idx] = 1

        # Accumulate confusion matrices
        accumulated_confusion_matrix += confusion_matrix

    # Step 3: Average the confusion matrix
    averaged_confusion_matrix = accumulated_confusion_matrix / num_subjects

    # Step 4: Get unique categories and their indices
    aggregated_categories = target_embeddings['category'].values
    unique_categories, category_indices = np.unique(aggregated_categories, return_index=True)

    # Step 5: Create the confusion matrix heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=averaged_confusion_matrix,
            colorscale='Reds',
            colorbar=dict(title='Average Classification'),
            xgap=0.05,
            ygap=0.05
        )
    )

    # Step 6: Add category borders and labels
    for i, (category, start_idx) in enumerate(zip(unique_categories, category_indices)):
        end_idx = category_indices[i + 1] - 1 if i + 1 < len(category_indices) else len(aggregated_categories) - 1

        # Add x-axis category bar
        fig.add_shape(
            type="rect",
            x0=start_idx - 0.5, x1=end_idx + 0.5,
            y0=-bar_size, y1=-0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=(start_idx + end_idx) / 2,
            y=-bar_size / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=90
        )

        # Add y-axis category bar
        fig.add_shape(
            type="rect",
            x0=-bar_size, x1=-0.5,
            y0=start_idx - 0.5, y1=end_idx + 0.5,
            line=dict(width=0),
            fillcolor=f"hsl({i * 360 / len(unique_categories)}, 50%, 75%)"
        )
        fig.add_annotation(
            x=-bar_size / 2,
            y=(start_idx + end_idx) / 2,
            text=category,
            showarrow=False,
            font=dict(size=12),
            textangle=0
        )

    # Step 7: Update layout for square plot and white background
    fig.update_layout(
        title=f'Average MSE Confusion Matrix (All Subjects)',
        font=dict(family="Editorial New Italic", size=16),
        width=600 + bar_size,
        height=600 + bar_size,
        xaxis=dict(title='Predicted Target Embeddings', tickvals=[], ticktext=[], scaleanchor='y'),
        yaxis=dict(title='True Trained Embeddings', tickvals=[], ticktext=[]),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Step 8: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Confusion_Matrices"
    os.makedirs(save_path, exist_ok=True)
    filename = f'average_{trained_type}_vs_{target_type}_mse_confusion_matrix.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Average confusion matrix saved: {os.path.join(save_path, filename)}")



def category_confusion_matrix_CS(base_path, subject_id, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    import pandas as pd

    # Step 1: Load and filter embeddings
    subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
    embeddings_df = load_embeddings(subject_folder)
    test_embeddings = embeddings_df[embeddings_df['split'] == 'test']
    
    # Step 2: Align target and trained embeddings by category and file path
    target_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == target_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    trained_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == trained_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    
    # Step 3: Compute cosine similarity
    target_matrix = np.stack(target_embeddings['embeddings'].values)
    trained_matrix = np.stack(trained_embeddings['embeddings'].values)
    similarity_matrix = cosine_similarity(trained_matrix, target_matrix)
    
    # Step 4: Determine predicted categories
    target_categories = target_embeddings['category'].values
    trained_categories = trained_embeddings['category'].values
    predicted_indices = np.argmax(similarity_matrix, axis=1)
    predicted_categories = target_categories[predicted_indices]
    
    # Step 5: Build category-level confusion matrix
    unique_categories = np.unique(target_categories)
    confusion_matrix = pd.DataFrame(
        np.zeros((len(unique_categories), len(unique_categories)), dtype=float),
        index=unique_categories,
        columns=unique_categories
    )
    
    for true_category, predicted_category in zip(trained_categories, predicted_categories):
        confusion_matrix.loc[true_category, predicted_category] += 1

    # Normalize the confusion matrix by true category counts
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100  # Convert to percentage
    
    # Step 6: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix.values,
            x=confusion_matrix.columns,
            y=confusion_matrix.index,
            colorscale='Blues',
            colorbar=dict(title='Percentage (%)'),
            zmin=0,
            zmax=100
        )
    )
    
    # Step 7: Add percentage text inside each cell
    for i, true_category in enumerate(confusion_matrix.index):
        for j, predicted_category in enumerate(confusion_matrix.columns):
            value = confusion_matrix.iloc[i, j]
            fig.add_annotation(
                x=predicted_category,
                y=true_category,
                text=f"{value:.1f}%",  # Format as percentage
                showarrow=False,
                font=dict(size=10, color="black" if value < 50 else "white")  # Adjust text color based on cell color
            )
    
    # Step 8: Update layout
    fig.update_layout(
        title=f'Subject {subject_id}',
        font=dict(family="Editorial New Italic", size=16),
        xaxis=dict(title='Predicted Categories', tickangle=-45),
        yaxis=dict(title='True Categories'),
        width=800 + bar_size,
        height=800 + bar_size,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Step 9: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Category_Confusion_Matrices"
    os.makedirs(save_path, exist_ok=True)
    filename = f'subject_{subject_id:02d}_{trained_type}_vs_{target_type}_category_confusion_matrix.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Normalized category-level confusion matrix saved: {os.path.join(save_path, filename)}")

def category_confusion_matrix_CS_all_subjects(base_path, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    import os
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # Step 1: Initialize variables for averaging
    accumulated_confusion_matrix = None
    num_subjects = 10

    # Step 2: Iterate over all subjects and compute category-level confusion matrices
    for subject_id in range(1, num_subjects + 1):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        test_embeddings = embeddings_df[embeddings_df['split'] == 'test']

        # Align target and trained embeddings by category and file path
        target_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == target_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )
        trained_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == trained_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )

        # Compute cosine similarity
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        similarity_matrix = cosine_similarity(trained_matrix, target_matrix)

        # Determine predicted categories
        target_categories = target_embeddings['category'].values
        trained_categories = trained_embeddings['category'].values
        predicted_indices = np.argmax(similarity_matrix, axis=1)
        predicted_categories = target_categories[predicted_indices]

        # Build category-level confusion matrix
        unique_categories = np.unique(target_categories)
        confusion_matrix = pd.DataFrame(
            np.zeros((len(unique_categories), len(unique_categories)), dtype=float),
            index=unique_categories,
            columns=unique_categories
        )

        for true_category, predicted_category in zip(trained_categories, predicted_categories):
            confusion_matrix.loc[true_category, predicted_category] += 1

        # Normalize the confusion matrix by true category counts
        confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0).fillna(0) * 100

        # Accumulate confusion matrices
        if accumulated_confusion_matrix is None:
            accumulated_confusion_matrix = confusion_matrix
        else:
            accumulated_confusion_matrix = accumulated_confusion_matrix.add(confusion_matrix, fill_value=0)

    # Step 3: Average the confusion matrix across subjects
    averaged_confusion_matrix = accumulated_confusion_matrix / num_subjects

    # Step 4: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=averaged_confusion_matrix.values,
            x=averaged_confusion_matrix.columns,
            y=averaged_confusion_matrix.index,
            colorscale='Blues',
            colorbar=dict(title='Percentage (%)'),
            zmin=0,
            zmax=100
        )
    )

    # Step 5: Add percentage text inside each cell
    for i, true_category in enumerate(averaged_confusion_matrix.index):
        for j, predicted_category in enumerate(averaged_confusion_matrix.columns):
            value = averaged_confusion_matrix.iloc[i, j]
            fig.add_annotation(
                x=predicted_category,
                y=true_category,
                text=f"{value:.1f}%",
                showarrow=False,
                font=dict(size=10, color="black" if value < 50 else "white")
            )

    # Step 6: Update layout
    fig.update_layout(
        title=f'Average Normalized Category-Level Confusion Matrix (All Subjects)',
        font=dict(family="Editorial New Italic", size=16),
        xaxis=dict(title='Predicted Categories', tickangle=-45),
        yaxis=dict(title='True Categories'),
        width=800 + bar_size,
        height=800 + bar_size,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Step 7: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Category_Confusion_Matrices"
    os.makedirs(save_path, exist_ok=True)
    filename = f'average_{trained_type}_vs_{target_type}_category_confusion_matrix.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Average normalized category-level confusion matrix saved: {os.path.join(save_path, filename)}")


def category_confusion_matrix_MSE(base_path, subject_id, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    import pandas as pd

    # Step 1: Load and filter embeddings
    subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
    embeddings_df = load_embeddings(subject_folder)
    test_embeddings = embeddings_df[embeddings_df['split'] == 'test']
    
    # Step 2: Align target and trained embeddings by category and file path
    target_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == target_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    trained_embeddings = (
        test_embeddings[test_embeddings['embedding_type'] == trained_type]
        .sort_values(by=['category', 'original_img_path'])
        .head(200)
    )
    
    # Step 3: Compute MSE
    target_matrix = np.stack(target_embeddings['embeddings'].values)
    trained_matrix = np.stack(trained_embeddings['embeddings'].values)
    mse_matrix = np.mean((trained_matrix[:, np.newaxis] - target_matrix) ** 2, axis=2)
    
    # Step 4: Determine predicted categories
    target_categories = target_embeddings['category'].values
    trained_categories = trained_embeddings['category'].values
    predicted_indices = np.argmin(mse_matrix, axis=1)
    predicted_categories = target_categories[predicted_indices]
    
    # Step 5: Build category-level confusion matrix
    unique_categories = np.unique(target_categories)
    confusion_matrix = pd.DataFrame(
        np.zeros((len(unique_categories), len(unique_categories)), dtype=float),
        index=unique_categories,
        columns=unique_categories
    )
    
    for true_category, predicted_category in zip(trained_categories, predicted_categories):
        confusion_matrix.loc[true_category, predicted_category] += 1

    # Normalize the confusion matrix by true category counts
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100  # Convert to percentage
    
    # Step 6: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix.values,
            x=confusion_matrix.columns,
            y=confusion_matrix.index,
            colorscale='Reds',  # Reds theme for MSE
            colorbar=dict(title='Percentage (%)'),
            zmin=0,
            zmax=100
        )
    )
    
    # Step 7: Add percentage text inside each cell
    for i, true_category in enumerate(confusion_matrix.index):
        for j, predicted_category in enumerate(confusion_matrix.columns):
            value = confusion_matrix.iloc[i, j]
            fig.add_annotation(
                x=predicted_category,
                y=true_category,
                text=f"{value:.1f}%",  # Format as percentage
                showarrow=False,
                font=dict(size=10, color="black" if value < 50 else "white")  # Adjust text color based on cell color
            )
    
    # Step 8: Update layout
    fig.update_layout(
        title=f'Subject {subject_id}',
        font=dict(family="Editorial New Italic", size=16),
        xaxis=dict(title='Predicted Categories', tickangle=-45),
        yaxis=dict(title='True Categories'),
        width=800 + bar_size,
        height=800 + bar_size,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Step 9: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Category_Confusion_Matrices"
    os.makedirs(save_path, exist_ok=True)
    filename = f'subject_{subject_id:02d}_{trained_type}_vs_{target_type}_mse_confusion_matrix.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Normalized MSE category-level confusion matrix saved: {os.path.join(save_path, filename)}")

def category_confusion_matrix_MSE_all_subjects(base_path, target_type='IMG', trained_type='ATMS', bar_size=75, scale=1):
    import os
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # Step 1: Initialize variables for averaging
    accumulated_confusion_matrix = None
    num_subjects = 10

    # Step 2: Iterate over all subjects and compute category-level confusion matrices
    for subject_id in range(1, num_subjects + 1):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        test_embeddings = embeddings_df[embeddings_df['split'] == 'test']

        # Align target and trained embeddings by category and file path
        target_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == target_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )
        trained_embeddings = (
            test_embeddings[test_embeddings['embedding_type'] == trained_type]
            .sort_values(by=['category', 'original_img_path'])
            .head(200)
        )

        # Compute MSE
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        mse_matrix = np.mean((trained_matrix[:, np.newaxis] - target_matrix) ** 2, axis=2)

        # Determine predicted categories
        target_categories = target_embeddings['category'].values
        trained_categories = trained_embeddings['category'].values
        predicted_indices = np.argmin(mse_matrix, axis=1)
        predicted_categories = target_categories[predicted_indices]

        # Build category-level confusion matrix
        unique_categories = np.unique(target_categories)
        confusion_matrix = pd.DataFrame(
            np.zeros((len(unique_categories), len(unique_categories)), dtype=float),
            index=unique_categories,
            columns=unique_categories
        )

        for true_category, predicted_category in zip(trained_categories, predicted_categories):
            confusion_matrix.loc[true_category, predicted_category] += 1

        # Normalize the confusion matrix by true category counts
        confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0).fillna(0) * 100

        # Accumulate confusion matrices
        if accumulated_confusion_matrix is None:
            accumulated_confusion_matrix = confusion_matrix
        else:
            accumulated_confusion_matrix = accumulated_confusion_matrix.add(confusion_matrix, fill_value=0)

    # Step 3: Average the confusion matrix across subjects
    averaged_confusion_matrix = accumulated_confusion_matrix / num_subjects

    # Step 4: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=averaged_confusion_matrix.values,
            x=averaged_confusion_matrix.columns,
            y=averaged_confusion_matrix.index,
            colorscale='Reds',
            colorbar=dict(title='Percentage (%)'),
            zmin=0,
            zmax=100
        )
    )

    # Step 5: Add percentage text inside each cell
    for i, true_category in enumerate(averaged_confusion_matrix.index):
        for j, predicted_category in enumerate(averaged_confusion_matrix.columns):
            value = averaged_confusion_matrix.iloc[i, j]
            fig.add_annotation(
                x=predicted_category,
                y=true_category,
                text=f"{value:.1f}%",
                showarrow=False,
                font=dict(size=10, color="black" if value < 50 else "white")
            )

    # Step 6: Update layout
    fig.update_layout(
        title=f'Average Normalized MSE Category-Level Confusion Matrix (All Subjects)',
        font=dict(family="Editorial New Italic", size=16),
        xaxis=dict(title='Predicted Categories', tickangle=-45),
        yaxis=dict(title='True Categories'),
        width=800 + bar_size,
        height=800 + bar_size,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Step 7: Save as HTML
    save_path = f"/work3/s184984/repos/EEG_Image_decode/Generation/results/Category_Confusion_Matrices"
    os.makedirs(save_path, exist_ok=True)
    filename = f'average_{trained_type}_vs_{target_type}_mse_category_confusion_matrix.html'
    fig.write_html(os.path.join(save_path, filename))
    print(f"Average normalized MSE category-level confusion matrix saved: {os.path.join(save_path, filename)}")


import os
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from plotly.express.colors import qualitative
import random

def calculate_and_plot_scores(base_path, target_type='IMG', trained_type='ATMS', save_path=None, width=1500, height=400, scale=2, plot_stds=True, seed = 42):
    """
    Calculate accuracy metrics and plot the results for overall, category, and per-category accuracy.

    Parameters:
        base_path (str): Base path containing embeddings for all subjects.
        target_type (str): Type of the target embeddings (e.g., 'IMG').
        trained_type (str): Type of the trained embeddings (e.g., 'ATMS').
        save_path (str): Path to save the plot. If None, displays the plot.
        width (int): Width of the saved plot.
        height (int): Height of the saved plot.
        scale (int): Scale factor for the saved plot.
        plot_stds (bool): Whether to include standard deviations in the plot.
    """
    random.seed(seed)
    
    def load_subject_embeddings(subject_id):
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        embeddings_df = load_embeddings(subject_folder)
        return embeddings_df[embeddings_df['split'] == 'test']

    def calculate_metrics(subject_id):
        embeddings_df = load_subject_embeddings(subject_id)
        
        target_embeddings = (
            embeddings_df[embeddings_df['embedding_type'] == target_type]
            .sort_values(by='original_img_path')
            .head(200)
        )
        trained_embeddings = (
            embeddings_df[embeddings_df['embedding_type'] == trained_type]
            .sort_values(by='original_img_path')
            .head(200)
        )
        
        target_matrix = np.stack(target_embeddings['embeddings'].values)
        trained_matrix = np.stack(trained_embeddings['embeddings'].values)
        similarity_matrix = cosine_similarity(trained_matrix, target_matrix)
        
        correct_filepaths = list(target_embeddings['original_img_path'])
        target_categories = target_embeddings['category'].values
        trained_categories = trained_embeddings['category'].values
        
        accuracy = 0
        category_accuracy = 0
        per_category_counts = {category: {'correct': 0, 'total': 0} for category in np.unique(target_categories)}
        
        for i, row in enumerate(similarity_matrix):
            # Get the index of the most similar target embedding
            predicted_idx = np.argmax(row)
            predicted_filepath = correct_filepaths[predicted_idx]
            predicted_category = target_categories[predicted_idx]
            
            # Update overall accuracy
            if correct_filepaths[i] == predicted_filepath:
                accuracy += 1
            
            # Update category accuracy
            if target_categories[i] == predicted_category:
                category_accuracy += 1
            
            # Update per-category accuracy
            current_category = target_categories[i]
            per_category_counts[current_category]['total'] += 1
            if target_categories[i] == predicted_category:
                per_category_counts[current_category]['correct'] += 1
        
        total = len(correct_filepaths)
        per_category_accuracy = {
            category: counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            for category, counts in per_category_counts.items()
        }
        
        return {
            'accuracy': accuracy / total,
            'category_accuracy': category_accuracy / total,
            'per_category_accuracy': per_category_accuracy
        }

    # Collect metrics for all subjects
    subjects = [f'sub-{i:02d}' for i in range(1, 11)]
    subject_scores = {}
    for subject_id in range(1, 11):
        subject_scores[f'sub-{subject_id:02d}'] = calculate_metrics(subject_id)

    # Aggregate metrics across subjects
    categories = list(subject_scores['sub-01']['per_category_accuracy'].keys())
    overall_scores = {metric: [] for metric in ['accuracy', 'category_accuracy']}
    category_scores = {category: [] for category in categories}
    
    for subject in subjects:
        overall_scores['accuracy'].append(subject_scores[subject]['accuracy'])
        overall_scores['category_accuracy'].append(subject_scores[subject]['category_accuracy'])
        for category in categories:
            category_scores[category].append(subject_scores[subject]['per_category_accuracy'][category])

    # Compute statistics
    overall_stats = {
        metric: {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'min_subject': subjects[np.argmin(scores)],
            'max_subject': subjects[np.argmax(scores)]
        }
        for metric, scores in overall_scores.items()
    }
    category_stats = {
        category: {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'min_subject': subjects[np.argmin(scores)],
            'max_subject': subjects[np.argmax(scores)]
        }
        for category, scores in category_scores.items()
    }

    # Plot the results
    fig = go.Figure()
    subject_colors = {subject: qualitative.Plotly[i % len(qualitative.Plotly)] for i, subject in enumerate(subjects)}

    # Plot overall scores
    for metric, stats in overall_stats.items():
        fig.add_trace(
            go.Bar(
                x=[metric],
                y=[stats['mean']],
                name=f"{metric.capitalize()} Mean",
                error_y=dict(type='data', array=[stats['std']]) if plot_stds else None,
                marker_color='#636EFA',
                showlegend=False
            )
        )
        # Min/Max points
        fig.add_trace(
            go.Scatter(
                x=[metric],
                y=[stats['min']],
                mode='markers',
                marker=dict(color=subject_colors[stats['min_subject']], size=10),
                name=stats['min_subject'],
                legendgroup=stats['min_subject'],
                showlegend=False  # Legend managed separately
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[metric],
                y=[stats['max']],
                mode='markers',
                marker=dict(color=subject_colors[stats['max_subject']], size=10),
                name=stats['max_subject'],
                legendgroup=stats['max_subject'],
                showlegend=False  # Legend managed separately
            )
        )

    # Plot category scores
    for category, stats in category_stats.items():
        fig.add_trace(
            go.Bar(
                x=[category],
                y=[stats['mean']],
                name=f"{category} Mean",
                error_y=dict(type='data', array=[stats['std']]) if plot_stds else None,
                marker_color='#8C9EFA',
                showlegend=False
            )
        )
        # Min/Max points
        fig.add_trace(
            go.Scatter(
                x=[category],
                y=[stats['min']],
                mode='markers',
                marker=dict(color=subject_colors[stats['min_subject']], size=10),
                name=stats['min_subject'],
                legendgroup=stats['min_subject'],
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[category],
                y=[stats['max']],
                mode='markers',
                marker=dict(color=subject_colors[stats['max_subject']], size=10),
                name=stats['max_subject'],
                legendgroup=stats['max_subject'],
                showlegend=False
            )
        )

    # Add single legend for subjects
    for subject, color in subject_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, size=10),
                name=subject,
                legendgroup=subject,
                showlegend=True
            )
        )

    # Add vertical line
    fig.add_vline(
        x=1.5,
        line=dict(color="black", dash="dash"),
        annotation_text="Overall/Category Divider",
        annotation_position="top"
    )

    # Update layout
    fig.update_layout(
        title="Accuracies and Per-Category Analysis",
        xaxis=dict(title="Metrics / Categories", tickfont=dict(size=18)),
        yaxis=dict(title="Score", range=[0, 1.05], tickfont=dict(size=18)),
        font=dict(family="Editorial New Italic", size=16),
        barmode='group',
        legend=dict(
            title="Subjects",
            orientation="h",
            y=-0.3,
            x=0,
            traceorder="normal"
        ),
        margin=dict(b=200)
    )

    # Save or display the plot
    if save_path:
        pio.write_image(fig, save_path, width=width, height=height, scale=scale)
    else:
        fig.show()


# LBP AND COLOR

from PIL import Image
from skimage import color
from skimage.feature import local_binary_pattern
from scipy.spatial import distance
import cv2

def compute_lbp_histogram(image_np, P=8, R=1):
    """
    Computes LBP and histogram for an image.
    
    Parameters:
    - image_np: numpy array of the image (assumes RGB).
    - P: number of circularly symmetric neighbor set points (typically 8).
    - R: radius of circle (typically 1 for fine details).
    
    Returns:
    - lbp_hist: normalized histogram of LBP values.
    """
    # Convert image to grayscale
    gray_image = color.rgb2gray(image_np)
    
    # Compute LBP
    lbp = local_binary_pattern(gray_image, P, R, method="uniform")
    
    # Compute the histogram of LBP
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return lbp_hist

def compare_textures(img1, img2):

    lbp_hist1 = compute_lbp_histogram(img1)
    lbp_hist2 = compute_lbp_histogram(img2)

    # Compute correlation score
    # TODO: CHECK this
    correlation_score = 1 - distance.correlation(lbp_hist1, lbp_hist2)

    return correlation_score



# Colors: 
def compute_color_histogram(img):

    hist = {}
    for i, color in enumerate(['r', 'g', 'b']):
        hist[color] = cv2.calcHist([img], [i], None, [256], [0, 256])

    return hist

def compare_colors(img1, img2, measure):

    measures = {'corr': cv2.HISTCMP_CORREL, # higher = better
                'intersect': cv2.HISTCMP_INTERSECT, # higher = better
                'chi2': cv2.HISTCMP_CHISQR, # lower = better
                'bhat': cv2.HISTCMP_BHATTACHARYYA} # lower = better

    hist1 = compute_color_histogram(img1)
    hist2 = compute_color_histogram(img2)

    for color in hist1.keys():
        hist1[color] = cv2.normalize(hist1[color], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        hist2[color] = cv2.normalize(hist2[color], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

    scores = {color: cv2.compareHist(hist1[color], hist2[color], measures[measure]) for color in hist1.keys()}

    return scores


import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import random

def plot_images_and_metrics_combined(
    base_path,
    K=5,
    seed=42,
    save_path=None,
    image_size=(500, 500)
):
    """
    Create a combined plot of images and their associated metrics (MSE, Cosine, Color, LBP).

    Args:
        base_path (str): Path to embeddings and generated images.
        K (int): Number of images to sample for plotting.
        seed (int): Random seed for reproducibility.
        save_path (str): Path to save the combined plot.
        image_size (tuple): Target size for images (default: 500x500).
    """
    random.seed(seed)

    def load_subject_data(subject_id):
        """Load embeddings and paths for a specific subject."""
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        df = load_embeddings(subject_folder)

        target_paths = df[(df['split'] == 'test') & (df['embedding_type'] == 'IMG')]['original_img_path']
        target_embeddings = np.stack(df[(df['split'] == 'test') & (df['embedding_type'] == 'IMG')]['embeddings'])

        eeg_paths = df[(df['split'] == 'test') & (df['embedding_type'] == 'DIFFUSION')]['original_img_path']
        eeg_embeddings = np.stack(df[(df['split'] == 'test') & (df['embedding_type'] == 'DIFFUSION')]['embeddings'])

        return target_paths, target_embeddings, eeg_paths, eeg_embeddings

    def compute_metrics(eeg_embedding, target_embedding):
        """Compute metrics for a single subject."""
        mse = np.mean((eeg_embedding - target_embedding) ** 2)
        cosine = cosine_similarity(eeg_embedding[np.newaxis, :], target_embedding[np.newaxis, :])[0, 0]
        return mse, cosine

    # Sample K images
    all_subject_data = [load_subject_data(subject_id) for subject_id in range(1, 11)]
    target_paths = all_subject_data[0][0]
    sampled_indices = random.sample(range(len(target_paths)), K)

    # Create image grid with matplotlib
    fig, axes = plt.subplots(K, 11, figsize=(22, K * 2.5), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for row_idx, img_idx in enumerate(sampled_indices):
        target_path = target_paths.iloc[img_idx]
        target_img = cv2.imread(target_path)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.resize(target_img, image_size)
        axes[row_idx, 0].imshow(target_img)
        axes[row_idx, 0].axis('off')

        for col_idx, subject_data in enumerate(all_subject_data, start=1):
            reconstructed_path = os.path.join(
                base_path.replace('embeddings', 'generated_imgs').replace('EEG_encoder', ''),
                f'sub-{col_idx:02d}',
                os.path.basename(os.path.dirname(target_path)),
                os.path.basename(target_path)
            )
            reconstructed_img = cv2.imread(reconstructed_path)
            if reconstructed_img is not None:
                reconstructed_img = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB)
                reconstructed_img = cv2.resize(reconstructed_img, image_size)
                axes[row_idx, col_idx].imshow(reconstructed_img)
            axes[row_idx, col_idx].axis('off')

    # Save the image grid
    image_grid_path = os.path.join(save_path, f'image_grid_seed_{seed}.png')
    plt.tight_layout()
    plt.savefig(image_grid_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create metric bar plot with Plotly
    bar_fig = make_subplots(rows=K, cols=1, vertical_spacing=0.05)
    metric_colors = {
        'mse': '#FF4136',  # Red
        'cosine': '#0074D9',  # Blue
        'lbp': '#AAAAAA',  # Grey
        'color': '#2ECC40'  # Green
    }
    metric_labels = ['MSE', 'Cosine', 'LBP', 'Color']

    for row_idx, img_idx in enumerate(sampled_indices, start=1):
        metrics = {'mse': [], 'cosine': [], 'lbp': [], 'color': []}
        for subject_id, subject_data in enumerate(all_subject_data):
            eeg_embedding = subject_data[3][img_idx]
            target_embedding = subject_data[1][img_idx]
            mse, cosine = compute_metrics(eeg_embedding, target_embedding)
            metrics['mse'].append(mse)
            metrics['cosine'].append(cosine)
            # Placeholder: Replace with actual calculations for LBP and Color
            metrics['lbp'].append(random.random())
            metrics['color'].append(random.random())

        avg_metrics = [np.mean(metrics[metric]) for metric in metrics.keys()]
        std_metrics = [np.std(metrics[metric]) for metric in metrics.keys()]

        bar_fig.add_trace(
            go.Bar(
                y=metric_labels,  # Horizontal bars
                x=avg_metrics,
                error_x=dict(type='data', array=std_metrics),
                marker_color=[metric_colors[m] for m in metrics.keys()],
                orientation='h',
                showlegend=False
            ),
            row=row_idx, col=1
        )

    bar_fig.update_xaxes(range=[0, 1])  # Set x-axis range to [0, 1]
    bar_fig.update_layout(
        height=K * 250, width=800, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
        font=dict(size=20)  # Increase font size
    )

    # Save the bar plot
    bar_path = os.path.join(save_path, f'metrics_seed_{seed}.png')
    bar_fig.write_image(bar_path)

    # Combine the image grid and bar plot using PIL
    img1 = Image.open(image_grid_path)
    img2 = Image.open(bar_path)

    # Rescale img2 to match the height of img1
    img2_height = img1.height
    img2_width = int(img2.width * (img2_height / img2.height))
    img2 = img2.resize((img2_width, img2_height), Image.Resampling.LANCZOS)

    total_width = img1.width + img2.width
    max_height = img1.height

    combined_img = Image.new("RGB", (total_width, max_height), "white")
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))

    combined_path = os.path.join(save_path, f'combined_seed_{seed}.png')
    combined_img.save(combined_path)

    print(f"Combined plot saved at {combined_path}")


def calculate_and_plot_metrics(
    base_path, 
    save_path=None, 
    width=1500, 
    height=400, 
    scale=2, 
    shuffle_targets=False
):
    """
    Compute and plot metrics for cosine similarity, MSE, LBP, and color correlations.

    Args:
        base_path (str): Path to the embeddings directory.
        save_path (str): Path to save the plots.
        width (int): Plot width.
        height (int): Plot height.
        scale (int): Plot scaling factor.
        shuffle_targets (bool): Whether to shuffle target embeddings for baseline.
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.metrics.pairwise import cosine_similarity
    import random

    # Define subject colors
    subjects = [f'sub-{i:02d}' for i in range(1, 11)]
    subject_colors = {subject: qualitative.Plotly[i % len(qualitative.Plotly)] for i, subject in enumerate(subjects)}

    # Define color schemes for metrics
    metric_colors = {
        'cosine': ('#636EFA', '#8C9EFA'),
        'mse': ('#EF553B', '#F78B8B'),
        'lbp': ('#7F7F7F', '#B0B0B0'),
        'color': ('#2CA02C', '#98DF8A')
    }

    def load_subject_data(subject_id):
        """Load embeddings and paths for a specific subject."""
        subject_folder = os.path.join(base_path, f'sub-{subject_id:02d}')
        df = load_embeddings(subject_folder)

        target_paths = df[(df['split'] == 'test') & (df['embedding_type'] == 'IMG')]['original_img_path']
        target_embeddings = np.stack(df[(df['split'] == 'test') & (df['embedding_type'] == 'IMG')]['embeddings'])

        eeg_paths = df[(df['split'] == 'test') & (df['embedding_type'] == 'DIFFUSION')]['original_img_path']
        eeg_embeddings = np.stack(df[(df['split'] == 'test') & (df['embedding_type'] == 'DIFFUSION')]['embeddings'])

        categories = df[(df['split'] == 'test') & (df['embedding_type'] == 'IMG')]['category']

        return target_paths, target_embeddings, eeg_paths, eeg_embeddings, categories

    def calculate_subject_metrics(subject_id):
        """Calculate metrics for a specific subject."""
        target_paths, target_embeddings, eeg_paths, eeg_embeddings, categories = load_subject_data(subject_id)

        if shuffle_targets:
            target_embeddings = list(target_embeddings)  # Convert to list for shuffling
            target_paths = list(target_paths)  # Convert to list for shuffling
            random.shuffle(target_embeddings)
            random.shuffle(target_paths)
            target_embeddings = np.array(target_embeddings)  # Convert back to NumPy array
            target_paths = pd.Series(target_paths)  # Convert back to pandas Series

        # Cosine Similarity
        cosine_scores = [
            cosine_similarity(eeg[np.newaxis, :], target[np.newaxis, :])[0, 0]
            for eeg, target in zip(eeg_embeddings, target_embeddings)
        ]

        # Mean Squared Error (MSE)
        mse_scores = [np.mean((eeg - target) ** 2) for eeg, target in zip(eeg_embeddings, target_embeddings)]

        # LBP and Color Correlation
        lbp_scores, color_scores = [], []
        for eeg_path, target_path in zip(eeg_paths, target_paths):
            # Reconstructed image path
            sub_folder, img_folder = os.path.basename(os.path.dirname(target_path)), os.path.basename(target_path)
            reconstructed_path = os.path.join(
                base_path.replace('embeddings', 'generated_imgs').replace('EEG_encoder', ''),
                f'sub-{subject_id:02d}',
                sub_folder,
                img_folder
            )

            # Load images
            original_img = cv2.imread(target_path)
            reconstructed_img = cv2.imread(reconstructed_path)

            if original_img is None or reconstructed_img is None:
                continue

            # Color Correlation
            color_correlation = np.mean(list(compare_colors(original_img, reconstructed_img, 'corr').values()))
            color_scores.append(color_correlation)

            # Texture Correlation (LBP)
            correlation_score = compare_textures(original_img, reconstructed_img)
            lbp_scores.append(correlation_score)

        return {
            'cosine': cosine_scores,
            'mse': mse_scores,
            'lbp': lbp_scores,
            'color': color_scores,
            'categories': categories.values
        }

    def aggregate_metrics():
        """Aggregate metrics across all subjects."""
        metrics = {'cosine': [], 'mse': [], 'lbp': [], 'color': []}
        category_metrics = {'cosine': {}, 'mse': {}, 'lbp': {}, 'color': {}}

        for subject_id in range(1, 11):
            subject_results = calculate_subject_metrics(subject_id)

            # Overall scores
            for metric in metrics.keys():
                metrics[metric].append(np.mean(subject_results[metric]))

            # Per-category scores
            categories = subject_results['categories']
            for metric in category_metrics.keys():
                for category, score in zip(categories, subject_results[metric]):
                    if category not in category_metrics[metric]:
                        category_metrics[metric][category] = {}
                    if subjects[subject_id - 1] not in category_metrics[metric][category]:
                        category_metrics[metric][category][subjects[subject_id - 1]] = []
                    category_metrics[metric][category][subjects[subject_id - 1]].append(score)

        return metrics, category_metrics

    def calculate_stats(metrics, category_metrics):
        """Calculate statistics for metrics."""
        overall_stats = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'min_subject': subjects[np.argmin(scores)],
                'max_subject': subjects[np.argmax(scores)]
            }
            for metric, scores in metrics.items()
        }

        category_stats = {
            metric: {
                category: {
                    'mean': np.mean([np.mean(scores) for scores in subject_scores.values()]),
                    'std': np.std([np.mean(scores) for scores in subject_scores.values()]),
                    'min': np.min([np.mean(scores) for scores in subject_scores.values()]),
                    'max': np.max([np.mean(scores) for scores in subject_scores.values()]),
                    'min_subject': min(subject_scores.items(), key=lambda x: np.mean(x[1]))[0],
                    'max_subject': max(subject_scores.items(), key=lambda x: np.mean(x[1]))[0]
                }
                for category, subject_scores in category_metrics[metric].items()
            }
            for metric in category_metrics.keys()
        }

        return overall_stats, category_stats

    def plot_metrics(overall_stats, category_stats, metric_colors):
        """Plot metrics using Plotly."""
        for metric, (overall_color, category_color) in metric_colors.items():
            fig = go.Figure()

            # Overall bar
            stats = overall_stats[metric]
            fig.add_trace(
                go.Bar(
                    x=["Overall"],
                    y=[stats['mean']],
                    error_y=dict(type='data', array=[stats['std']]),
                    marker_color=overall_color,
                    name="Overall",
                    showlegend=False
                )
            )

            # Min/max points for overall
            fig.add_trace(
                go.Scatter(
                    x=["Overall"],
                    y=[stats['min']],
                    mode='markers',
                    marker=dict(color=subject_colors[stats['min_subject']], size=10),
                    name=stats['min_subject'],
                    legendgroup=stats['min_subject'],
                    showlegend=False
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=["Overall"],
                    y=[stats['max']],
                    mode='markers',
                    marker=dict(color=subject_colors[stats['max_subject']], size=10),
                    name=stats['max_subject'],
                    legendgroup=stats['max_subject'],
                    showlegend=False
                )
            )

            # Category bars
            for category, cat_stats in category_stats[metric].items():
                fig.add_trace(
                    go.Bar(
                        x=[category],
                        y=[cat_stats['mean']],
                        error_y=dict(type='data', array=[cat_stats['std']]),
                        marker_color=category_color,
                        name=category,
                        showlegend=False
                    )
                )

                # Min/max points for categories
                fig.add_trace(
                    go.Scatter(
                        x=[category],
                        y=[cat_stats['min']],
                        mode='markers',
                        marker=dict(color=subject_colors[cat_stats['min_subject']], size=10),
                        name=cat_stats['min_subject'],
                        legendgroup=cat_stats['min_subject'],
                        showlegend=False
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[category],
                        y=[cat_stats['max']],
                        mode='markers',
                        marker=dict(color=subject_colors[cat_stats['max_subject']], size=10),
                        name=cat_stats['max_subject'],
                        legendgroup=cat_stats['max_subject'],
                        showlegend=False
                    )
                )

            # Subject legend
            for subject, color in subject_colors.items():
                fig.add_trace(
                    go.Scatter(
                        x=[None],  # Dummy trace
                        y=[None],
                        mode='markers',
                        marker=dict(color=color, size=10),
                        name=subject,
                        legendgroup=subject,
                        showlegend=True
                    )
                )

            # Layout
            fig.update_layout(
                # title=f"{metric.capitalize()} Histogram",
                xaxis_title="Metrics / Categories",
                yaxis_title="Score",
                barmode='group',
                width=width,
                height=height,
                legend=dict(title="Subjects", orientation="h", y=-0.3, x=0)
            )

            # Save or show
            if save_path:
                save_name = f"baseline_{metric}_histogram.png" if shuffle_targets else f"{metric}_histogram.png"
                fig.write_image(os.path.join(save_path, save_name), width=width, height=height, scale=scale)
            else:
                fig.show()

    # Main computation and plotting
    metrics, category_metrics = aggregate_metrics()
    overall_stats, category_stats = calculate_stats(metrics, category_metrics)
    plot_metrics(overall_stats, category_stats, metric_colors)

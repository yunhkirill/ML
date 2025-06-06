import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import load_protonet_conv
from utils import extract_sample, read_images


def visualize_predictions(model, test_x, test_y, n_way, n_support, n_query, num_episodes=5, save_path="results/prediction_visualisation"):
    """
    Visualizes the predictions from the protonet model and saves them to disk.

    Args:
        model: trained model
        test_x (np.array): images of testing set
        test_y (np.array): labels of testing set
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        num_episodes (int): number of episodes to visualize
        save_path (str, optional): directory to save visualizations. Defaults to 'data'.
    """
    model.eval()
    os.makedirs(save_path, exist_ok=True)

    for episode in range(num_episodes):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)

        with torch.no_grad():
            loss, output = model.set_forward_loss(sample)

        sample_images = sample['images'].cpu()
        y_hat = output['y_hat'].cpu().numpy()

        fig, axes = plt.subplots(n_way, n_support + n_query,
                                 figsize=(2*(n_support + n_query), 2*n_way),
                                 gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

        fig.suptitle(f"Episode {episode + 1} - Accuracy: {output['acc']:.4f}", fontsize=16, y=0.98)

        plt.figtext(0.25, 0.95, "Support Set", fontsize=12, ha='center')
        plt.figtext(0.75, 0.95, "Query Set", fontsize=12, ha='center')

        for i in range(n_way):
            plt.figtext(0.01, (1.0 - (i + 0.5) / n_way), f"Class {i}", fontsize=12, ha='left', va='center')

        query_indices = np.arange(n_way * n_query).reshape(n_way, n_query)

        for i in range(n_way):
            for j in range(n_support + n_query):
                img = sample_images[i, j].permute(1, 2, 0).numpy()

                ax = axes[i, j]

                if img.max() > 1:
                    img = img / 255.0

                ax.imshow(img)

                ax.set_xticks([])
                ax.set_yticks([])

                if j >= n_support:
                    query_idx = j - n_support
                    flat_idx = query_indices[i, query_idx]
                    pred_class = y_hat[flat_idx]

                    if pred_class == i:
                        for spine in ax.spines.values():
                            spine.set_edgecolor('green')
                            spine.set_linewidth(2)
                    else:
                        for spine in ax.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(2)

                        ax.set_title(f"→ {int(pred_class)}", fontsize=10, color='red')

        for i in range(n_way):
            axes[i, n_support-1].axvline(x=1.0, color='black', linestyle='-', linewidth=1, alpha=0.3)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.1, hspace=0.1)

        save_file = os.path.join(save_path, f"episode_{episode + 1}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    model = load_protonet_conv(
        x_dim=(3, 28, 28),
        hid_dim=64,
        z_dim=64,
    )

    model_path = "data/protonet_best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=model.device)

    model.load_state_dict(checkpoint)

    model.eval()

    n_way = 5
    n_support = 5
    n_query = 5
    test_x, test_y = read_images('data/images_evaluation')

    num_episodes = 3

    visualize_predictions(model, test_x, test_y, n_way, n_support, n_query, num_episodes)
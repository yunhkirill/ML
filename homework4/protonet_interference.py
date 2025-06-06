import os
import torch
from tqdm import tqdm

from model import load_protonet_conv
from utils import extract_sample, read_images, display_sample


def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    """
    Tests the protonet
    Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
    """
    running_loss = 0.0
    running_acc = 0.0

    for episode in tqdm(range(test_episode), desc="Testing"):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output['loss']
        running_acc += output['acc']

    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))

    return avg_loss, avg_acc


if __name__ == "__main__":
    model = load_protonet_conv(
        x_dim=(3, 28, 28),
        hid_dim=64,
        z_dim=64,
    )

    model_path = "data/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=model.device)

    model.load_state_dict(checkpoint)

    model.eval()

    n_way = 5
    n_support = 5
    n_query = 5
    test_x, test_y = read_images('data/images_evaluation')

    test_episode = 1000

    test(model, test_x, test_y, n_way, n_support, n_query, test_episode)

    my_sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
    display_sample(my_sample['images'])
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, images, labels, class_names, transform_augment=None):
        self.images = images
        self.labels = labels
        self.class_names = class_names
        self.transform_augment = transform_augment

        self.transform_basic = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = transforms.ToPILImage()(image)

        if self.transform_augment:
            image = self.transform_augment(image)

        image = self.transform_basic(image)

        return image, label
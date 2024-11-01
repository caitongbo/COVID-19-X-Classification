from torchvision import transforms
# from torchtoolbox.transform import Cutout

def train_transforms(width, height):
    trans_list = [
        transforms.Resize((height, width)),
        # Cutout(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=20,
                                    translate=(0.15, 0.15),
                                    scale=(0.8, 1.2),
                                    shear=5)], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        # transforms.ToTensor()
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    return transforms.Compose(trans_list)


def val_transforms(width, height):
    trans_list = [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
                # transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    return transforms.Compose(trans_list)

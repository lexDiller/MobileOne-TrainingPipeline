from dataloader import PinsFaceDataset
from train_validate import train_model
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2)
    ),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

train_dataset = PinsFaceDataset(
    data_directory='/home/moo/PycharmProjects/clsf_yolo/105_classes_pins_dataset/train',
    transform=test_transform
)

test_dataset = PinsFaceDataset(
    data_directory='/home/moo/PycharmProjects/clsf_yolo/105_classes_pins_dataset/val',  # используем val как test
    transform=test_transform
)

model, history = train_model(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    num_classes=len(train_dataset.classes),
    num_epochs=100
)
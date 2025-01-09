import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import numpy as np
from model import MobileOneWithTriplet
from dataloader import PinsFaceDataset


def run_inference(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Running inference"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, logits = model(inputs)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Получаем метрики для каждого класса
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=test_loader.dataset.classes,
        digits=4,
        output_dict=True
    )

    print("\nPer-class Metrics:")
    for class_name in test_loader.dataset.classes:
        print(f"\nClass: {class_name}")
        print(f"Precision: {class_report[class_name]['precision']:.4f}")
        print(f"Recall: {class_report[class_name]['recall']:.4f}")
        print(f"F1-score: {class_report[class_name]['f1-score']:.4f}")
        print(f"Support: {class_report[class_name]['support']}")

    # Выводим общие метрики
    print("\nOverall Metrics:")
    print(f"Accuracy: {class_report['accuracy']:.4f}")
    print(f"Macro Precision: {class_report['macro avg']['precision']:.4f}")
    print(f"Macro Recall: {class_report['macro avg']['recall']:.4f}")
    print(f"Macro F1-score: {class_report['macro avg']['f1-score']:.4f}")
    print(f"\nWeighted Precision: {class_report['weighted avg']['precision']:.4f}")
    print(f"Weighted Recall: {class_report['weighted avg']['recall']:.4f}")
    print(f"Weighted F1-score: {class_report['weighted avg']['f1-score']:.4f}")

    # Сохраняем результаты в файл
    with open('test_results.txt', 'w') as f:
        f.write("Per-class Metrics:\n")
        for class_name in test_loader.dataset.classes:
            f.write(f"\nClass: {class_name}\n")
            f.write(f"Precision: {class_report[class_name]['precision']:.4f}\n")
            f.write(f"Recall: {class_report[class_name]['recall']:.4f}\n")
            f.write(f"F1-score: {class_report[class_name]['f1-score']:.4f}\n")
            f.write(f"Support: {class_report[class_name]['support']}\n")

        f.write("\nOverall Metrics:\n")
        f.write(f"Accuracy: {class_report['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {class_report['macro avg']['precision']:.4f}\n")
        f.write(f"Macro Recall: {class_report['macro avg']['recall']:.4f}\n")
        f.write(f"Macro F1-score: {class_report['macro avg']['f1-score']:.4f}\n")
        f.write(f"\nWeighted Precision: {class_report['weighted avg']['precision']:.4f}\n")
        f.write(f"Weighted Recall: {class_report['weighted avg']['recall']:.4f}\n")
        f.write(f"Weighted F1-score: {class_report['weighted avg']['f1-score']:.4f}\n")

    return class_report


if __name__ == "__main__":
    data_directory = "/home/moo/PycharmProjects/clsf_yolo/105_classes_pins_dataset/val"
    model_path = "apple-mb1.pt"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model = MobileOneWithTriplet(num_classes=105)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PinsFaceDataset(data_directory=data_directory, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    metrics = run_inference(model, test_loader, device)

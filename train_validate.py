from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from loss import CombinedLoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import pandas as pd
from model import MobileOneWithTriplet



def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        features, logits = model(inputs)
        loss = criterion(features, logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        pbar.set_postfix({'loss': running_loss / total, 'acc': f'{acc:.2f}%'})

    return running_loss / len(train_loader), acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)

            features, logits = model(inputs)
            loss = criterion(features, logits, labels)

            running_loss += loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return running_loss / len(val_loader), acc


def evaluate_metrics(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Evaluation'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, logits = model(inputs)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Вычисляем метрики
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Вычисляем метрики для каждого класса
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                                                             average=None)

    class_metrics = pd.DataFrame({
        'Class': val_loader.dataset.classes,
        'Precision': precision_per_class,
        'Recall': recall_per_class,
        'F1-score': f1_per_class
    })

    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nPer-class Metrics:")
    print(class_metrics.to_string())

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics
    }


def train_model(train_dataset, test_dataset, num_classes, num_epochs=100):
    batch_size = 32
    learning_rate = 1e-3
    margin = 0.5
    triplet_weight = 0.8

    pretrained_path = '/home/moo/Downloads/mobileone_s4_unfused.pth.tar'

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MobileOneWithTriplet(
        num_classes=num_classes,
        pretrained_path=pretrained_path
    ).to(device)

    criterion = CombinedLoss(margin=margin, triplet_weight=triplet_weight)
    optimizer = optim.AdamW([
        {'params': model.base.parameters(), 'lr': learning_rate * 0.1},  # меньший lr для предобученных слоев
        {'params': model.classifier.parameters(), 'lr': learning_rate}  # больший lr для нового классификатора
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0
    best_model_path = 'apple-mb1.pt'

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_metrics': []
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if (epoch + 1) % 5 == 0:
            print("\nCalculating test metrics:")
            test_metrics = evaluate_metrics(model, test_loader, device)
            history['test_metrics'].append({
                'epoch': epoch + 1,
                **test_metrics
            })

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')

    print("\nEvaluating best model on test set:")
    model.load_state_dict(torch.load(best_model_path))
    final_metrics = evaluate_metrics(model, test_loader, device)

    final_metrics['class_metrics'].to_csv('final_test_metrics.csv', index=False)

    return model, history
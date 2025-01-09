import torch.nn as nn
import torch
from mobileone import mobileone


class MobileOneWithTriplet(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        # Инициализируем базовую модель с 1000 классами (как в ImageNet)
        self.base = mobileone(variant='s4', num_classes=1000)

        # Загружаем предобученные веса, если путь предоставлен
        if pretrained_path:
            checkpoint = torch.load(pretrained_path)
            # Некоторые ключи в state_dict могут отличаться, поэтому используем strict=False
            self.base.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint,
                                      strict=False)
            print("Loaded pretrained weights from:", pretrained_path)

        # Получаем размерность признаков из последнего слоя
        in_features = self.base.linear.in_features

        # Создаем новый классификатор для нашего количества классов
        self.classifier = nn.Linear(in_features, num_classes)

        # Заменяем оригинальный классификатор на Identity
        self.base.linear = nn.Identity()

    def forward(self, x):
        # Получаем features
        features = self.base(x)

        # Получаем логиты через новый классификатор
        logits = self.classifier(features)

        return features, logits


# class MobileOneWithTriplet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.base = timm.create_model('mobileone_s4', pretrained=False)
#
#         # Получаем размерность признаков из последнего слоя
#         in_features = self.base.head.fc.in_features
#
#         # Заменяем классификатор
#         self.base.head.fc = nn.Linear(in_features, num_classes)
#
#     def forward(self, x):
#         # Получаем признаки до последнего слоя
#         x = self.base.forward_features(x)
#
#         # Global Average Pooling
#         features = x.mean([2, 3])
#
#         # Получаем логиты через классификатор
#         logits = self.base.head.fc(features)
#
#         return features, logits

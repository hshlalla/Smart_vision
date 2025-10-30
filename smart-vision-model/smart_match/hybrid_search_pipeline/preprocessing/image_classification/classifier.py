import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet152, ResNet152_Weights
from sklearn.metrics import f1_score

# 디바이스 설정 (GPU 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 전처리 및 증강
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.8, 1.2))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 데이터 경로 설정 및 데이터셋 로드
data_dir = "../../data/binary_label_classifier"

def create_custom_imagefolder_structure(data_dir, transform):
    class CustomImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            classes = []
            for upper_dir in os.listdir(directory):
                upper_path = os.path.join(directory, upper_dir)
                if os.path.isdir(upper_path):
                    for lower_dir in os.listdir(upper_path):
                        lower_path = os.path.join(upper_path, lower_dir)
                        if os.path.isdir(lower_path):
                            classes.append(f"{upper_dir}/{lower_dir}")
            class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
            return classes, class_to_idx

    return CustomImageFolder(data_dir, transform=transform)

full_dataset = create_custom_imagefolder_structure(data_dir, transform=data_transforms['train'])

# 데이터셋 분할 (train:val = 90:10)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 하위 카테고리를 반영한 클래스 이름 확인
class_names = full_dataset.classes
print(f"Total classes (subcategories): {len(class_names)}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# DataLoader 생성
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

# ResNet152 모델 불러오기 및 수정
model = resnet152(weights=ResNet152_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))  # 클래스 수에 맞게 출력 레이어 수정
model = model.to(device)

# 손실 함수 및 최적화 함수 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 학습 루프
def train_model_with_metrics(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # F1 Score 계산을 위한 저장
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # F1 스코어 계산
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')  # 가중 평균 F1 스코어

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

            if phase == 'val' and epoch_f1 > best_f1:  # F1 기준으로 Best Model 저장
                best_f1 = epoch_f1
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f"Best val Acc: {best_acc:.4f}, Best val F1: {best_f1:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# 틀린 예측 확인 함수
def log_wrong_predictions(model, dataloader):
    model.eval()
    wrong_predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_classes = torch.topk(probs, 2, dim=1)

            for i in range(inputs.size(0)):
                if top_classes[i, 0] != labels[i]:
                    wrong_predictions.append({
                        'input': inputs[i].cpu(),
                        'predicted_1st': top_classes[i, 0].item(),
                        'predicted_2nd': top_classes[i, 1].item(),
                        'actual': labels[i].item(),
                        'prob_1st': top_probs[i, 0].item(),
                        'prob_2nd': top_probs[i, 1].item()
                    })
    return wrong_predictions

# 학습 시작
model = train_model_with_metrics(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=100)

# Validation 틀린 예측 확인
data_val_loader = dataloaders['val']
wrong_predictions = log_wrong_predictions(model, data_val_loader)
print(f"Number of wrong predictions: {len(wrong_predictions)}")

# 모델 저장
torch.save(model.state_dict(), "resnet152_lazer_model.pth")

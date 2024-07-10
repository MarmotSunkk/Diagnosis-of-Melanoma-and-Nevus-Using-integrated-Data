
from PIL import Image
import torch.nn.functional as F
from net import vgg16
import os
from torchvision.datasets import ImageFolder
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# directory = r'D:\Group_project\CNN\test_1'
# file_list = []
# for file in os.listdir(directory):
#     file_list.append(os.path.join(directory, file))
#     # print(file_list)



# Data pre-processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the size required for VGG16
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardisation
])

# Load dataset
dataset = ImageFolder(root=r'.\train', transform=transform)

# Getting images and tags
images = [image[0] for image in dataset.imgs]
labels = [label for _, label in dataset.imgs]

# 5 cross validation with StratifiedKFold
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=41)

# Initialise the list of accuracies and F1-scores
accuracies = []
f1_scores = []

for fold, (train_indices, test_indices) in enumerate(skf.split(images, labels)):
    print(f"Fold {fold + 1}:")

    # Create training and validation sets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

    '''加载网络'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = vgg16()
    model = torch.load(r".\model\nm_vgg16_classifier_epoch31.pth", map_location=device)  # Input of trained resultant weights
    net.load_state_dict(model)  # Model import
    net.eval()  # Set to speculative mode

    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    # Calculate accuracy and F1-score
    print(y_true)
    print(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    print('F1-score: {:.2f}'.format(f1))

    accuracies.append(accuracy)
    f1_scores.append(f1)

# Calculate average accuracy and F1-score
avg_accuracy = sum(accuracies) / len(accuracies)
avg_f1 = sum(f1_scores) / len(f1_scores)
print('\nAverage Accuracy: {:.2f}%'.format(avg_accuracy * 100))
print('Average F1-score: {:.2f}'.format(avg_f1))
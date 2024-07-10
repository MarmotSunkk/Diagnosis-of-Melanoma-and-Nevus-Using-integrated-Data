import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from net import vgg16
import os

directory = r'D:\Group_project\vggmast-master\vggmast-master\test_1'
file_list = []
for file in os.listdir(directory):
    file_list.append(os.path.join(directory, file))
    #print(file_list)
    '''Handling image size'''
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    '''Loading Network'''
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net =vgg16()# Input network
        model=torch.load(r".\model\DogandCat4.pth",map_location=device)# Trained result weights input
    net.load_state_dict(model)#模型导入
    net.eval()#设置为推测模式

# 加载测试集数据
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():  # 在评估阶段关闭梯度计算
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Test Accuracy: {:.2f}%'.format(accuracy))
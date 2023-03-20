import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

def get_season(img):
    model = models.resnet18(pretrained=True)
    num_classes = 4
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # load saved state dictionary
    state_dict = torch.load('best_model_resnet_ALL.pth', map_location=torch.device('cpu'))

    # create a new model with the correct architecture
    new_model = models.resnet18(pretrained=True)
    new_model.fc = nn.Linear(in_features, num_classes)
    new_model.load_state_dict(state_dict)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(img).convert('RGB')
    image = transform(image).unsqueeze(0)

    new_model.eval()

    with torch.no_grad():
        output = new_model(image)
    pred_index = output.argmax().item()
    print("Decided color: ",pred_index)
    return pred_index

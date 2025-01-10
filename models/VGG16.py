
import torch
import torchvision



def get_VGG16():
    """
    Get the VGG16 model with the number of classes specified in the configuration file.
    """
    model = torchvision.models.vgg16(pretrained=True)
    # Freeze the feature extraction part
    for param in model.parameters():
        param.requires_grad = False
    # Replace the head
    # model.fc = torch.nn.Linear(model.fc.in_features, 4) # corresponding to the 4 coordinates of the bounding box
    model.classifier = torch.nn.Sequential(
        *list(model.classifier[:6].children()) + [
            torch.nn.Linear(model.classifier[6].in_features, 4),
            torch.nn.Sigmoid()
        ]
    ) # corresponding to the 4 coordinates of the bounding box, and a sigmoid activation function to normalise the values between 0 and 1
    return model








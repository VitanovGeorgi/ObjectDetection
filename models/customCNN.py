import torch


def get_customCNN():
    """
    Get the custom CNN model with the number of classes specified in the configuration file.
    """

    return CustomCNN()


class CustomCNN(torch.nn.Module):
    """
    Custom CNN model.
    This is just a generic, small model, inspired by the VGG16 model. Mainly here to show how to create a custom model.
    And reuse the functions from VGG16.
    """
    def __init__(self) -> None:
        super(CustomCNN, self).__init__()
        
        # solely separated for readability
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1), # input (3, 224, 224), output (64, 224, 224)
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), # output (64, 224, 224)
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # output (64, 112, 112)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 112 * 112, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 4),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
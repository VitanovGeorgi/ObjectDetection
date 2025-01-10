"""
Utility methods for constructing losses.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig



def create_loss(config: DictConfig) -> torch.nn.Module:
    """
    Create the loss function based on the configuration file.

    Args:
        config (DictConfig): The configuration dictionary.

    Returns:
        torch.nn.Module: The loss function.
    """

    if config.training.loss == 'fasterrcnn_resnet50_fpn':
        loss = Resnet50Loss()
    elif config.training.loss == 'mse':
        loss = torch.nn.MSELoss()
    elif config.training.loss == 'iou':
        loss = IOU()
    else:
        raise ValueError(f"Loss {config.training.loss} not recognized.")

    return loss



class Resnet50Loss(torch.nn.Module):
    """
        Frankly we don't need this, we could have just as easily extract the loss_box_reg in the training loop.
    We're only doing this for completeness, in case we need to add more losses in the future, or simply to keep the code consistent.
    """
    def __init__(self) -> None:
        super(Resnet50Loss, self).__init__()

    def forward(self, outputs: dict) -> torch.Tensor:
        # return outputs['loss_classifier'], outputs['loss_box_reg'], outputs['loss_objectness'], outputs['loss_rpn_box_reg']
        return outputs['loss_box_reg']



class VGG16Loss(torch.nn.Module):
    """
        Frankly we don't need this, we could have just as easily extract the loss_box_reg in the training loop.
    We're only doing this for completeness, in case we need to add more losses in the future, or simply to keep the code consistent.
    """
    def __init__(self) -> None:
        super(VGG16Loss, self).__init__()

    def forward(self, outputs, targets) -> torch.Tensor:
        # return outputs['loss_classifier'], outputs['loss_box_reg'], outputs['loss_objectness'], outputs['loss_rpn_box_reg']
        return torch.nn.MSELoss(outputs, targets)


class IOU(torch.nn.Module):
    """
        Intersection over Union loss. Taken from 
        https://github.com/LukeDitria/pytorch_tutorials/blob/main/section08_detection/solutions/Pytorch1_Bounding_Box_Detection_Solutions.ipynb
    """
    def __init__(self) -> None:
        super(IOU, self).__init__()

    def _xyxy_to_xywh(self, boxes):
        """
            For all the models we have, they expect the bounding box to be of the format x1, y1, x2, y2. But to calculate the IOU, we need the bounding box
        to be of the format x, y, width, height. This function converts the bounding box from the former to the latter. We don't make any verification that 
        the bounding boxes provided are actually in this format, i.e. were we to have a model which expects them to be of the latter format, here we wouldn't 
        catch this. This is left to the user to mitigate.
        """

        new_boxes = torch.cat(
            (
                boxes[:, 0:1],
                boxes[:, 1:2],
                boxes[:, 2:3] - boxes[:, 0:1],
                boxes[:, 3:4] - boxes[:, 1:2]
            ), dim=1
        )

        return new_boxes


    def forward(self, outputs, targets) -> torch.Tensor:
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = torch.cat((outputs[:, 0:1], targets[:, 0:1]), 1).max(dim=1)[0].unsqueeze(1)
        yA = torch.cat((outputs[:, 1:2], targets[:, 1:2]), 1).max(dim=1)[0].unsqueeze(1)
        xB = torch.cat((outputs[:, 2:3], targets[:, 2:3]), 1).min(dim=1)[0].unsqueeze(1)
        yB = torch.cat((outputs[:, 3:4], targets[:, 3:4]), 1).min(dim=1)[0].unsqueeze(1)

        # Compute the area of intersection rectangle
        x_len = F.relu(xB - xA)
        y_len = F.relu(yB - yA)
        # Negative area means no overlap
        interArea = x_len * y_len
        # If you don't have xyhw values, calculate areas like this
        w1 = (outputs[:, 0:1] - outputs[:, 2:3]).abs()
        h1 = (outputs[:, 1:2] - outputs[:, 3:4]).abs()

        w2 = (targets[:, 0:1] - targets[:, 2:3]).abs()
        h2 = (targets[:, 1:2] - targets[:, 3:4]).abs()

        area1 = w1 * h1
        area2 = w2 * h2

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / (area1 + area2 - interArea + 1e-5)

        return iou

"""
    Utility functions for training.
"""

from typing import Optional

import torch
import torchvision
from torchmetrics import Metric
from tqdm import tqdm
from omegaconf import DictConfig


def train_one_epoch_resnet50(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    metric: Metric,
    device: torch.device,
    epoch: int,
    logger,
    rank: int = Optional[int]
) -> None:
    """
    Train the model for one epoch.

    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the training on.
        epoch (int): The epoch number.
        loss_fn (torch.nn.Module): The loss function.
        logger: The logger object.
    """
    model.train()
    metric.reset()

    for _, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch + 1}", leave=True, position=0)):
        images, image_names, targets = batch
        # Move to device
        images_input = [img.to(device, non_blocking=True) for img in images]
        targets_input = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        # Forward pass
        outputs = model(images_input, targets_input)
        # Backward pass
        optimizer.zero_grad()
        # Compute the loss
        # loss = sum(loss for loss in outputs.values())
        loss = loss_fn(outputs) 
        # epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Update the metric
        metric.update(loss.item())
    # Calculate and log metrics
    if rank is not None:
        if rank == 0:
            metrics = metric.compute()
            logger.log_metrics(metrics, step=epoch + 1)
    else:
        metrics = metric.compute()
        logger.log_metrics(metrics, step=epoch + 1)
    metric.reset()


def validate_one_epoch_resnet50(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    metric: Metric,
    device: torch.device,
    epoch: int,
    logger,
    rank: int = Optional[int]
) -> None:
    """
    Validate the model for one epoch.

    Args:
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        model (torch.nn.Module): The model.
        device (torch.device): The device to run the training on.
        epoch (int): The epoch number.
        loss_fn (torch.nn.Module): The loss function.
        logger: The logger object.
    """
    model.eval()

    predictions = {}
    total_images = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(val_loader, desc=f"Validation epoch {epoch + 1}", leave=True, position=0)):
            images, image_names, targets = batch
            # Move to device
            images_input = [img.to(device, non_blocking=True) for img in images]
            targets_input = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            # Forward pass
            outputs = model(images_input)
            # Compute the loss
            for prediction, img_name in zip(outputs, image_names):
                predictions[img_name] = {
                    "boxes": prediction["boxes"][0],
                    "labels": prediction["labels"][0],
                    "scores": prediction["scores"][0],
                }
            total_images += len(images)
    
    logger.log_metrics(predictions, step=epoch + 1)


def test_one_epoch_resnet50(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    metric: Metric,
    device: torch.device,
    epoch: int,
    logger,
    rank: int = Optional[int]
) -> None:
    """
    Test the model for one epoch. Well the model is tested in a single epoch, the naming is just for consistency.
    We already receive scores and predictions from the model, so we don't need to compute the loss. The output
    from the model is of the form:
    ({
        "boxes": torch.FloatTensor[N, 4], # N is the number of predicted boxes, 4 coordinates for each box
        "labels": torch.Int64Tensor[N], # labels for each predicted box, in our case should all be 1s
        "scores": torch.Tensor[N], # scores for each predicted box, i.e. the confidence of the model in the prediction
    }).
    Now the thing is, the model will predict multiple bounding boxes, and score them each. Because it isn"t of importance
    now to aggregate over the multiple predictions, we can just take the highest confidence score as the prediction.

    Args:
        test_loader (torch.utils.data.DataLoader): The test data loader.
        model (torch.nn.Module): The model.
        device (torch.device): The device to run the training on.
        logger: The logger object.
    """
    model.eval()
    predictions = {
        "boxes": torch.tensor([]),
        "labels": torch.tensor([]),
        "scores": torch.tensor([]),
    }
    total_images = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader, desc=f"Testing", leave=True, position=0)):
            images, image_names, targets = batch
            # Move to device
            images_input = [img.to(device, non_blocking=True) for img in images]
            targets_input = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            # Forward pass
            outputs = model(images_input)
            # Extract the predicitons
            for prediction, img_name in zip(outputs, image_names):
                predictions[img_name] = {
                    "boxes": prediction["boxes"][0],
                    "labels": prediction["labels"][0],
                    "scores": prediction["scores"][0],
                }
            total_images += len(images)

    logger.log_metrics(predictions)


def train_one_epoch_vgg16(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    metric: Metric,
    device: torch.device,
    epoch: int,
    logger,
    rank: int = Optional[int]
) -> None:
    """
        Train the model for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            model (torch.nn.Module): The model.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (torch.device): The device to run the training on.
            epoch (int): The epoch number.
            loss_fn (torch.nn.Module): The loss function.
            logger: The logger object.
    """

    model.train()
    metric.reset()

    for _, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch + 1}", leave=True, position=0)):
        images, image_names, targets = batch
        # Forward pass
        outputs = model(images.to(device, non_blocking=True))
        # Backward pass
        optimizer.zero_grad()
        # Compute the loss
        # loss = sum(loss for loss in outputs.values())
        loss = loss_fn(outputs, targets.to(device, non_blocking=True)) 
        # epoch_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        # Update the metric
        metric.update(loss.mean().item(), type="train")
    # Calculate and log metrics
    if rank is not None:
        if rank == 0:
            metrics = metric.compute(type="train")
            logger.log_metrics(metrics, step=epoch + 1)
    else:
        metrics = metric.compute(type="train")
        logger.log_metrics(metrics, step=epoch + 1)


def validate_one_epoch_vgg16(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    metric: Metric,
    device: torch.device,
    epoch: int,
    logger,
    rank: int = Optional[int]
) -> None:
    """
        Validate the model for one epoch.

        Args:
            val_loader (torch.utils.data.DataLoader): The validation data loader.
            model (torch.nn.Module): The model.
            device (torch.device): The device to run the training on.
            epoch (int): The epoch number.
            loss_fn (torch.nn.Module): The loss function.
            logger: The logger object.
    """

    model.eval()
    metric.reset()

    with torch.no_grad():
        for _, batch in enumerate(tqdm(val_loader, desc=f"Validation epoch {epoch + 1}", leave=True, position=0)):
            images, image_names, targets = batch
            # Forward pass
            outputs = model(images.to(device, non_blocking=True))
            # Compute the loss
            loss = loss_fn(outputs, targets.to(device, non_blocking=True)) 
            # epoch_loss += loss.item()
            # Update the metric
            metric.update(loss.mean().item(), type="val")
    # Calculate and log metrics        
    if rank is not None:
        if rank == 0:
            metrics = metric.compute(type="val")
            logger.log_metrics(metrics, step=epoch + 1)
    else:
        metrics = metric.compute(type="val")
        logger.log_metrics(metrics, step=epoch + 1)


def test_one_epoch_vgg16(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    metric: Metric,
    device: torch.device,
    epoch: int,
    logger,
    rank: int = Optional[int]
) -> None:
    """
        Test the model for one epoch. 
        
        Args:
            model (torch.nn.Module): The model.
            test_loader (torch.utils.data.DataLoader): The test data loader.
            loss_fn (torch.nn.Module): The loss function.
            metric: The metric object.
            device (torch.device): The device to run the training on.
            epoch (int): The epoch number.
            logger: The logger object.

    """
    model.eval()
    metric.reset()

    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader, desc=f"Test epoch {epoch + 1}", leave=True, position=0)):
            images, image_names, targets = batch
            # Forward pass
            outputs = model(images.to(device, non_blocking=True))
            # Compute the loss
            loss = loss_fn(outputs, targets.to(device, non_blocking=True)) 
            # epoch_loss += loss.item()
            # Update the metric
            metric.update(loss.mean().item(), type="test")
    # Calculate and log metrics
    if rank is not None:
        if rank == 0:
            metrics = metric.compute(type="test")
            logger.log_metrics(metrics, step=epoch)
    else:
        metrics = metric.compute(type="test")
        logger.log_metrics(metrics, step=epoch)


def get_train_one_epoch(model) -> callable:
    """
    Get the train_one_epoch function for the model.

    Now most of these functions are similar to each other, and we could have placed them all in a single function, with some if statements. 
    But for fasterrcnn_resnet50_fpn, we are have implemented a different loss strategy then the other models, which isn't contained in the loss_fn
    (frankly the choices were either implement it from scratch - I didn't have the time, or just play with what is saved there). So for the purposes
    of this model, and modularity, we will keep them separate. This applies for validate_one_epoch and test_one_epoch as well.
    """
    if model == "fasterrcnn_resnet50_fpn":
        return train_one_epoch_resnet50
    elif model == "vgg16" or model == "custom_cnn":
        return train_one_epoch_vgg16
    else:
        raise NotImplementedError(f"Model {model} not implemented!")

def get_validate_one_epoch(model) -> callable:
    """
    Get the validate_one_epoch function for the model.
    """
    if model == "fasterrcnn_resnet50_fpn":
        return validate_one_epoch_resnet50
    elif model == "vgg16" or model == "custom_cnn":
        return validate_one_epoch_vgg16
    else:
        raise NotImplementedError(f"Model {model} not implemented!")

def get_test_one_epoch(model) -> callable:
    """
    Get the test_one_epoch function for the model.
    Test functions are similar (almost identical) to the validation ones
    """
    if model == "fasterrcnn_resnet50_fpn":
        return test_one_epoch_resnet50
    elif model == "vgg16" or model == "custom_cnn":
        return test_one_epoch_vgg16
    else:
        raise NotImplementedError(f"Model {model} not implemented!")






def create_optimizer(config: DictConfig, model) -> torch.optim.Optimizer:
    """
    Parse the configuration file and return a optimizer object to update the model parameters.
    """
    assert config.optimizer.optimizer in [
        "sgd",
        "adam",
    ], "Only SGD and Adam optimizers are available!"

    optim_params = [
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "lr": config.optimizer.lr,
            "weight_decay": config.optimizer.weight_decay,
        }
    ]

    if config.optimizer.optimizer == "sgd":
        return torch.optim.SGD(optim_params)
    elif config.optimizer.optimizer == "adam":
        return torch.optim.Adam(optim_params)
    

class CustomMetrics(Metric):
    """
    Custom metrics class. Only done so it's easier to add more metrics in the future.
    """

    def __init__(self, device):
        super().__init__()
        self.add_state("train_loss", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")
        self.add_state("val_loss", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")
        self.add_state("test_loss", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")

    def update(self, loss, type: str = "train"):
        if type == "train":
            self.train_loss += loss
        elif type == "val":
            self.val_loss += loss
        elif type == "test":
            self.test_loss += loss
        else:
            raise ValueError(f"Type {type} not recognized.")
        
    def compute(self, type: str = "train"):
        if type == "train":
            return {
                "train_loss": self.train_loss,
            }
        elif type == "val":
            return {
                "val_loss": self.val_loss,
            }
        elif type == "test":
            return {
                "test_loss": self.test_loss,
            }
    



from pathlib import Path
import pdb

import torch
import torch.optim as optim
from tqdm import tqdm

from utils.utils import reset_random_seeds
from utils.logger import set_logger_paths, WandBLogger
from utils.data import get_data_loaders
from utils.training import create_optimizer, CustomMetrics, \
      get_train_one_epoch, get_validate_one_epoch, get_test_one_epoch 
      

from models.models import create_model
from models.losses import create_loss






def train(cfg):
    """
    Run the experiments for the Robotics Engineer Intern - AI Defect Detection - Flyability project. This method will set up the device,
    correct paths, initialize tracking, generate the dataset, train the model, and evaluate it.

    Args:
        cfg (dict): The configuration dictionary.
    """

    # --------------------------
    # Set up the device
    # --------------------------

    # Reproducibility
    gen_random_seed = reset_random_seeds(cfg['random_seed'])

    # Device
    # torch.cuda.set_device(cfg.distributed.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Additional info when using cuda
    if device.type == 'cuda':
        print(f'Using {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU')
    
    # Set paths
    experiment_path, experiment_name = set_logger_paths(cfg)
    
    # WandB
    logger = WandBLogger(cfg, experiment_name)


    # --------------------------
    # Prepare the data loaders
    # --------------------------

    train_loader, val_loader, test_loader = get_data_loaders(cfg, gen_random_seed)


    # --------------------------
    # Initialize the model
    # --------------------------

    # Initialize a pre-trained model
    model = create_model(cfg).to(device)

    # load model, if it already exists
    if cfg.model.load_model_path != "":
        model.load_state_dict(torch.load(cfg.model.load_model_path, weights_only=True, map_location=device))
    elif (cfg.model.load_model_path == "") and cfg.inference:
        print("No model to load for inference. Please provide a model to load. Inference will be run on a randomly initialized model.")
        # raise ValueError("No model to load for inference.")

    model.to(device)

    loss_fn = create_loss(cfg)
    metrics = CustomMetrics(device=device)

    # --------------------------
    # Training the model
    # --------------------------

    num_epochs = cfg.training.epochs

    if not cfg.inference:

        optimizer = create_optimizer(cfg, model) # torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.optimizer.decrease_every,
            gamma=1 / cfg.optimizer.lr_divisor,
        )

        train_one_epoch = get_train_one_epoch(cfg.model.model)
        validate_one_epoch = get_validate_one_epoch(cfg.model.model)

        for epoch in range(num_epochs):
            """
                We will have the training and validation steps, encapsulated in a function, i.e. train_one_epoch and validate_one_epoch.
            By doing so, at each iteration of the training loop, we can have the training done, but at each validate_per_epoch, we
            will have the validation done. This way, we can have the model validated in a as many epochs as we wanted, in a single loop.
                Also epoch == 0 means 0 % cfg.training.validate_per_epoch == 0, so we will have the validation done at the first epoch, 
            so we just do epoch + 1 for validation.
            """
            print(f"Epoch {epoch + 1}")

            if cfg.distributed.use_distributed:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)
            
            train_one_epoch(
                model, train_loader, optimizer, loss_fn, metrics, device, epoch, logger, cfg.distributed.local_rank if cfg.distributed.use_distributed else None
            )
            if (epoch + 1) % cfg.training.validate_per_epoch == 0:
                validate_one_epoch(
                    model, val_loader, loss_fn, metrics, device, epoch, logger, cfg.distributed.local_rank if cfg.distributed.use_distributed else None
                )
            lr_scheduler.step()

    # --------------------------
    # Test the model
    # --------------------------
    
    test_one_epoch = get_test_one_epoch(cfg.model.model)

    test_one_epoch(
        model, test_loader, loss_fn, metrics, device, num_epochs, logger, cfg.distributed.local_rank if cfg.distributed.use_distributed else None
    )

    # save the model
    if cfg.save_model:
        model_save_path = Path(experiment_path) / f"{experiment_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"\nTRAINING FINISHED, MODEL SAVED AS {model_save_path}!", flush=True)
    else:
        print("\nTRAINING FINISHED!", flush=True)
    
    if cfg.demo:
        print("DEMO MODE ACTIVATED! END OF TRAINING!")
        return model, test_loader, metrics

    x = 0


















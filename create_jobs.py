import os
import yaml  # type: ignore
import itertools
from pathlib import Path
import copy
from typing import Any
import argparse


def load_yaml(file_path: Path | str) -> Any:
    """Load YAML file and return dictionary."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, file_path: Path) -> None:
    """Save dictionary as YAML file."""
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def find_hyperparameter_lists(data: dict) -> dict:
    """
    Recursively find all list parameters in the YAML structure.
    Returns a dictionary with their paths and values.
    """
    hyper_params = {}

    def recurse(current_data: Any, current_path: Any = []) -> Any:
        if isinstance(current_data, dict):
            for key, value in current_data.items():
                new_path = current_path + [key]
                recurse(value, new_path)
        elif isinstance(current_data, list):
            # Only consider it a hyperparameter if it's a list of numbers or strings
            # MAYBE TODO: if its a list of lists dont touch it (for indexes parameters)
            if all(isinstance(x, (int, float, str)) for x in current_data):
                path_str = ".".join(current_path)
                hyper_params[path_str] = current_data

    recurse(data)
    return hyper_params


def set_value_by_path(data: dict, path: str, value: Any) -> None:
    """Set a value in nested dictionary using dot notation path."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def generate_yaml_combinations(
    parent_yaml_path: Path | str, output_dir: Path | str = "generated_yamls"
) -> int:
    """
    Generate multiple YAML files based on hyperparameter combinations.
    """
    # Load parent YAML
    config = load_yaml(parent_yaml_path)

    # Find all hyperparameters (lists)
    hyper_params = find_hyperparameter_lists(config)

    # Generate all combinations of hyperparameters
    param_names = list(hyper_params.keys())
    param_values = [hyper_params[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    # Create output directory if it doesn't exist

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_subdir = Path(output_dir) / "configs"
    output_subdir.mkdir(exist_ok=True, parents=True)

    # Generate a YAML file for each combination
    for i, combo in enumerate(combinations, 1):
        # Create a deep copy of the original config
        new_config = copy.deepcopy(config)

        # Set the values for this combination
        for param_name, param_value in zip(param_names, combo):
            set_value_by_path(new_config, param_name, param_value)

        # Generate filename based on the parameters
        filename_parts = []
        for param_name, param_value in zip(param_names, combo):
            param_short_name = param_name.split(".")[-1]
            filename_parts.append(f"{param_short_name}-{param_value}")

        filename = f"config_{i}.yaml"
        save_yaml(new_config, output_subdir / filename)

    return len(combinations)



# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate from parent YAML, children YAMLs for gridsearch"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grid.yaml",
        help="Relative path to the parent YAML configuration file",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="grid",
        help="Relative path to folder to store the children YAML configuration files",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vgg16",
        help="Model name to be run (please pick froom the provided yaml files, as it'll be used to load the model)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="probe_images",
        help="Dataset to be used (we only have probe_images for now).",
    )
    
    args = parser.parse_args()
    # num_generated = generate_yaml_combinations(args.config, args.output_dir)
    num_generated = generate_yaml_combinations(
        args.config, args.output_dir
    )

    print(
        f"Generated {num_generated} YAML files with different hyperparameter combinations."
    )

    # Directory where the bash scripts will be created
    script_dir = f"{args.output_dir}/scripts"
    config_dir = f"{args.output_dir}/configs"
    
    # Ensure the output directory exists
    os.makedirs(script_dir, exist_ok=True)

    # Iterate over all files in the configs directory
    for config_file in os.listdir(config_dir):
        # Check if the file is a YAML file
        if config_file.endswith(".yaml"):
            # Create a corresponding bash script
            script_name = f"run_{config_file.replace('.yaml', '.sh')}"
            script_path = os.path.join(script_dir, script_name)

            # Write the bash script content
            with open(script_path, "w") as script:
                # command = f"python main.py --config-name {os.path.join(config_dir, config_file)} +model={args.model} +data={args.data}"
                command = f"python main.py --config {os.path.join(config_dir, config_file)}"
                script.write(f"#!/bin/bash\n{command}\n")

            # Make the script executable
            os.chmod(script_path, 0o755)

    print("Be aware that once the scripts are generated, you have to delete them manually. Otherwise, if this script is rerun \
        it may only overwrite part of the existing scripts, leaving the rest, depending on how many new scripts it needs to generate! \
          You can simply delete the folders inside the grid directory to start fresh.")
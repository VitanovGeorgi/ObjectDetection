# CV Internship Project

Welcome to the CV Internship Project! This project is designed to help you build and enhance your curriculum vitae (CV) for internship applications.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This project aims to provide a structured and professional template for creating a CV. It includes various sections that are commonly required by employers and helps you present your skills and experiences effectively.

## Features

- Professional CV template
- Easy to customize
- Sections for personal information, education, work experience, skills, and more
- Clean and modern design

## Installation

To get started with this project, clone the repository to your local machine:

```bash
git clone git@github.com:VitanovGeorgi/ObjectDetection.git
```

Navigate to the project directory:

```bash
cd CV_internship
```

## Usage

First, set up your environment. It is recommended to use `conda` for managing dependencies and environments, as it ensures consistency and ease of use.

To execute a single instance of the model, you can either modify the `configs/default.yaml` file to suit your requirements or run the following command:

```bash
python main.py --config configs/default.yaml
```

```bash
python main.py --config {your config file}
```


To create multiple jobs for hyperparameter tuning, place all the varying parameters in an array in the `configs/grid.yaml` file. Then, run the following command to generate the jobs:

```bash
python create_jobs.py
```

Once the jobs are created, execute them by running:

```bash
bash ./scripts/run_jobs.sh
```

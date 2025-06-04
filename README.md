# 🎲 dice_rl_TU_Vienna

This repository contains implementations of various _offline, behavior-agnostic Off-Policy Evaluation (OffPE)_ algorithms using _Stationary Distribution Correction Estimation (DICE)_. Developed at __TU Vienna__, these methods are designed to integrate with a wide range of _Reinforcement Learning (RL)_ applications.

## 📦 Installation

We recommend using a `conda` or `venv` environment to manage dependencies.

### Using Conda

```
# Create a new conda environment
conda create -n env_name python=3.11.9
conda activate env_name

# Install essential libraries via conda
conda install pandas=2.2.3 matplotlib=3.10.3 scipy=1.15.2 tqdm=4.67.1 pyarrow=20.0.0

# Install required Python packages
pip install --user --force-reinstall tf-nightly==2.20.0
pip install --user --force-reinstall tf-keras-nightly==2.20.0
pip install --user --force-reinstall tfp-nightly==0.26.0
pip install --user --force-reinstall dm-reverb-nightly==0.15.0
pip install --user --force-reinstall tf-agents-nightly==0.20.0
pip install torch==2.6.0
pip install d3rlpy==2.8.1
```

### Using venv

```
# Create a new venv environment with python 3.11.9 using macos
python3 -m venv env_name
source env_name/bin/activate  # Linux/macOS
# OR
env_name\Scripts\activate     # Windows

# Upgrade pip inside the environment
pip install --upgrade pip

# Install packages using pip
pip install pandas==2.2.3 matplotlib==3.10.3 scipy==1.15.2 tqdm==4.67.1 pyarrow==20.0.0
pip install tf-nightly==2.20.0 tf-keras-nightly==2.20.0 tfp-nightly==0.26.0 dm-reverb-nightly==0.15.0 tf-agents-nightly==0.20.0 torch==2.6.0 d3rlpy==2.8.1
```

## 📁 Project Structure

Clone this library into your project folder or make it a _GitHub submodule_.
Then you can use it inside a `script.py` or `notebook.ipynb` by calling `from dice_rl_TU_Vienna.submodule_path import something`.

```
your_project_folder/
├── dice_rl_TU_Vienna/
├── script.py
├── notebook.ipynb
└── ...
```

## 🚀 Features
- Implements several behavior-agnostic OffPE algorithms, particularly DICE-based
- Modular codebase for integration into existing RL pipelines
- Based on `TensorFlow` and `PyTorch`
- Easily extensible for different environments

## 📚 Documentation

You can find in-depth API documentation, class descriptions, and usage examples on the [GitHub Wiki](https://github.com/Reinforcement-Learning-TU-Vienna/dice_rl_TU_Vienna/wiki).

## 🧪 Example Usage

This library has been used in the project [dice_rl_sepsis](https://github.com/Reinforcement-Learning-TU-Vienna/dice_rl_sepsis), which provides code accompanying my paper:
_Evaluating Reinforcement-Learning-based Sepsis Treatments via Tabular and Continuous Stationary Distribution Correction Estimation_.

Feel free to explore that repo for a concrete example of how to apply the DICE algorithms to a real-world healthcare RL problem.

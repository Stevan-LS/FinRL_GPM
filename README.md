# FinRL_GPM - Investment Portfolio Optimization with Reinforcement Learning

![Portfolio Performance](plots/result_GPM.jpg)

## Overview

FinRL_GPM is an AI-driven investment portfolio manager that uses Generalized Policy Mirroring (GPM) reinforcement learning techniques to optimize portfolio allocation across multiple assets. The model learns optimal investment strategies by interacting with financial market environments and adapting its decision-making process to maximize returns while managing risk.

## Key Features

- **Reinforcement Learning Framework**: Uses FinRL (Financial Reinforcement Learning) to train investment policies
- **Generalized Policy Mirroring**: Implements GPM algorithm for improved policy optimization
- **Portfolio Optimization**: Automatically allocates investments across different assets
- **Performance Evaluation**: Includes tools to evaluate and visualize model performance

## Results

The trained model demonstrates significant portfolio growth compared to baseline strategies:

## Project Structure

- `train.py`: Main script for training the reinforcement learning model
- `app.py`: Application interface for using the trained model
- `dataProcessing.py`: Handles financial data preprocessing
- `FinRL_GPM_Demo.ipynb`: Jupyter notebook demonstrating the model's capabilities
- `policy_GPM.pt`: Pre-trained policy model weights
- `plots/`: Directory containing performance visualization results

## Getting Started

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Training a Model

To train a new model:

```bash
python train.py
```

### Testing Performance

To evaluate model performance:

```bash
python app_test.py
```

### Demo

For a comprehensive demonstration of the model's capabilities, see the `FinRL_GPM_Demo.ipynb` notebook.

## Technical Approach

This project combines financial market data analysis with deep reinforcement learning techniques. The agent learns to make investment decisions by balancing risk and reward across multiple market conditions. The Generalized Policy Mirroring approach enhances traditional reinforcement learning by improving policy convergence and stability.

## License

This project is available for educational and research purposes.

## Acknowledgments

This project builds upon the FinRL framework and incorporates Generalized Policy Mirroring techniques for enhanced performance.

Stevan Le Stanc
# AI Practitioner Quick Samples

This sample workspace helps you review core AI Practitioner topics with runnable examples and Docker support.

## What is included

- `train_mnist.py`: simple PyTorch model for MNIST classification. Covers network architecture, activation/loss choices, hyperparameters, initialization, optimization, gradient flow, and training diagnostics.
- `fairness_explainability.py`: synthetic fairness and explainability demo using scikit-learn. Covers protected attributes, fairness metrics, and model-agnostic explanation fallbacks.
- `mlops_pipeline.py`: a lightweight pipeline example showing data preparation, training, evaluation, experiment tracking, and model versioning.
- `Dockerfile`: containerizes the sample environment.
- `docker-compose.yml`: runs the training demo in a container.
- `requirements.txt`: dependencies for quick setup.

## Quick start

```bash
cd "${PWD}"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** If you encounter NumPy compatibility issues, ensure NumPy < 2.0 is installed:
```bash
pip install --upgrade "numpy<2.0.0"
```

### Run the MNIST training sample

```bash
python train_mnist.py --epochs 2 --batch-size 128 --lr 0.01
```

### Run the MLOps pipeline demo

```bash
python mlops_pipeline.py
```

### Run the fairness and explainability demo

```bash
python fairness_explainability.py
```

### Build and run with Docker

```bash
docker build -t ai-practitioner-sample .
docker run --rm ai-practitioner-sample
```

Or start with Docker Compose:

```bash
docker compose up --build
```

## Learning objective mapping

### Deep Learning Fundamentals

- `train_mnist.py` shows: model architecture, activation functions, loss selection, optimizer updates, initialization, scheduler, and gradient monitoring.
- Training notes explain how data flows through layers and how changing hyperparameters impacts learning.

### MLOps Fundamentals

- `mlops_pipeline.py` demonstrates a reproducible pipeline with deterministic seeding, experiment tracking, model saving, and staged validation.
- Version control is discussed in the pipeline comments and artifact naming.

### RAI Technical Reliability and Explainability

- `fairness_explainability.py` generates a biased synthetic dataset, computes group fairness metrics, and produces simple explanations.
- It also highlights ethical considerations for PII, human oversight, and trade-offs between fairness and accuracy.

### Docker Fundamentals

- `Dockerfile` uses lightweight `python:3.12-slim`, installs dependencies, copies source, and runs training.
- `docker-compose.yml` shows a multi-container-style configuration for local experimentation.

## Notes

- This is a learning sandbox, not a production pipeline.
- Use the scripts as conversation starters for exam topics and as code-based review examples.

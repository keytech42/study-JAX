# JAX Learning Project

This project is a workspace for learning JAX, focusing on high-performance numerical computing and building machine learning models like Transformers.

## Project Structure

- `mnist.py`: A script to load and visualize the MNIST dataset using JAX-compatible data pipelines.
- `transformer.py`: Implementation of Transformer architectures in JAX.
- `docs/`: Detailed documentation and learning notes.

## Quick Start

This project uses `uv` for dependency management.

```bash
# Sync dependencies
uv sync

# Run MNIST visualization
python mnist.py
```

## Documentation

For more detailed information, please refer to the following guides:

1.  [**Environment & Dependency Guide**](./docs/02_environment_and_dependencies.md): Explains the specific package versions (Colab baseline) and why they are pinned.
2.  [**Self-Attention Deep Dive**](./docs/01_self_attention_deep_dive.md): Conceptual notes on attention mechanisms.

---
*Note: This environment is optimized to match Google Colab's stable package stack.*

# Environment Setup & Dependency Guide

This document explains the rationale behind the specific package versions used in this project, particularly to ensure compatibility between JAX, TensorFlow, and `tensorflow-datasets` (TFDS).

## The Colab Baseline

The dependencies in this project (`uv.lock`, `pyproject.toml`) are strictly pinned to match the environment provided by Google Colab (as of our baseline). 

Why is this important? Local environments often pull the very latest sub-patch versions of libraries, which can lead to unexpected "Dependency Hell" where one library demands a version that breaks another. Colab environments use pre-tested, stable combinations.

## Key Dependencies & Roles

| Package | Role in Project | Compatibility Notes |
| :--- | :--- | :--- |
| **`jax` / `jaxlib`** | Core ML framework | Requires modern Numpy versions. We are pinned to `0.7.2`. |
| **`tensorflow`** | Data pipelines (`tf.data`) | Pinned to `2.20.0`. It relies heavily on `protobuf` for internal data serialization and communication. |
| **`tensorflow-datasets`** | Downloading datasets | Pinned to `4.9.9`. Provides a clean API for standard datasets like MNIST. |
| **`protobuf`** | Data Serialization | **CRITICAL:** Pinned strictly to `5.29.6`. See "Troubleshooting" below. |
| **`importlib_resources`** | File/Resource loading | Needed internally by TFDS to read checksums during dataset downloads. Pinned to `7.1.0`. |
| **`matplotlib`** | Visualization | Used to plot and verify the dataset images. |

## Troubleshooting Compatibility Issues

### 1. Protobuf `AttributeError`

**Symptom:**
```python
AttributeError: 'FieldDescriptor' object has no attribute 'label'
```

**Cause:**
`tensorflow-datasets` relies on older metadata schemas that expect certain attributes (like `label`) to exist in the `protobuf` library. If a newer `protobuf` version (e.g., `7.x`) is installed, these internal C++ structures differ, causing the script to crash upon loading the dataset info.

**Solution:**
Ensure `protobuf` is exactly `5.29.6` (the Colab baseline). With this specific version, the C++ backend and the TFDS Python frontend are perfectly compatible, and no additional environment variables (like `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) are required.

### 2. ModuleNotFoundError for `importlib_resources`

**Symptom:**
```python
ModuleNotFoundError: No module named 'importlib_resources'
```

**Cause:**
TFDS uses the `etils` library, which expects this package to read internal data (like download checksums). While newer Python versions have `importlib.resources` built-in, the external package is still required by the specific version of TFDS we use.

**Solution:**
Ensure `importlib_resources==7.1.0` is installed.

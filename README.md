# htm_py

A lightweight, pure Python implementation of Hierarchical Temporal Memory (HTM) for anomaly detection and entropy analysis of streaming numerical data. Designed for future-proof, production-ready use — no C++ build tools required.

## Goals

- Recreate the essential functionality of `htm.core` in Python 3+
- Validate outputs against historical Numenta benchmarks (e.g., NAB)
- Support a flexible config-based modeling interface
- Measure both anomaly scores and model ambiguity via prediction count

## Key Features

- Scalar + datetime encoders
- Spatial Pooler & Temporal Memory
- Combined `HTMModel` object for end-to-end streaming
- YAML configuration support for pipeline control
- Compatible with `htm_wl_demo` and similar real-time systems

## Installation

```bash
git clone https://github.com/gotham29/htm_py.git
cd htm_py
pip install .
```

## Usage

Example:

```python
from htm_py import HTMModel

model = HTMModel(config_path="my_model_config.yaml")
model.update(input_dict)
```

## License

MIT License

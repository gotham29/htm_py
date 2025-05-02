# htm_py

A pure-Python, production-ready Hierarchical Temporal Memory (HTM) library.

## Overview

`htm_py` implements key components of HTM theory:
- **Encoders** (RDSE and DateEncoder)
- **Spatial Pooler**
- **Temporal Memory**
- **Anomaly Score**
- **Anomaly Likelihood**
- **Prediction Count**

The library is engineered for:
- Faithfulness to Numenta’s NAB benchmark
- Python 3.x compatibility
- No C++ dependencies
- Lightweight, modular, and highly extensible

## Installation

```bash
pip install -r requirements.txt
```

(Requirements are minimal — mainly `numpy`, `scipy`, and `pytest`.)

## How to Run

Run NAB comparison:

```bash
python nab_tm_runner.py
```

Run full unit tests:

```bash
pytest tests
```

## Repository Structure

```
htm_py/
    connections.py
    date_encoder.py
    rdse_encoder.py
    spatial_pooler.py
    temporal_memory.py
    htm_model.py
tests/
    test_connections.py
    test_temporal_memory.py
    test_encoders.py
    test_htm_model.py
data/
    art_daily_jumpsup.csv
nab_tm_runner.py
requirements.txt
pytest.ini
README.md
```

## Notes

- `htm_py` matches Numenta NAB outputs **to machine precision** (anomaly scores & likelihoods).
- Designed to be easily extendable for production use cases.
- 100% Python, no need for htm.core, pycapnp, or old nupic bindings.

---

Built with ❤️ for robust, biologically inspired AI systems.

# StarVLA + LIBERO Evaluation Pipeline

This module provides a standardized evaluation pipeline for running **StarVLA policies on the LIBERO benchmark** and recording evaluation metrics in a clean, reproducible format.

The goal of this component is to make evaluation easy, reproducible, and comparable across different checkpoints and experiments.

---

# Overview

This evaluation system provides:

- A **clean evaluation API**
- **Standardized metrics format**
- **Automatic logging of results (JSON / CSV)**
- A **command line interface (CLI)** for running evaluations
- Compatibility with **official StarVLA checkpoints**

This module is designed to work with the **LIBERO evaluation environment** and **StarVLA models**.

---

# Project Structure
```
evaluation/
│
├── api.py # Clean evaluation API
├── metrics.py # Standardized metrics schema
├── logger.py # Result logging utilities
└── run_eval.py # CLI entry point
```

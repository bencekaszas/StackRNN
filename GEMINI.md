# Gemini Assistant Project Memory: StackRNN

This document provides a comprehensive overview of the StackRNN project to serve as a persistent context for AI-assisted development sessions.

## Project Overview

This is a Python-based research project investigating the out-of-distribution (OOD) generalization capabilities of memory-augmented Recurrent Neural Networks (RNNs). We are comparing how different architectures learn algorithmic tasks and generalize to inputs longer than those seen during training.

*   **Core Task:** Bit-string reversal using an **Encoder-Decoder** architecture.
*   **Technologies:** JAX, Flax, Optax, Matplotlib, Seaborn.
*   **Key Insight:** Soft-stack training with **Hard Stack Inference** (discrete actions at test time) allows perfect algorithmic generalization by eliminating cumulative pointer drift.

## Development Log & Recent Breakthroughs

1.  **Encoder-Decoder Refactor:** Successfully moved away from simple auto-regressive next-token prediction to a proper Encoder-Decoder setup. This allows the model to "plan" the reversal after reading the full input.
2.  **Diagnostic Suite:** Implemented a robust visualization suite (`visualize.py`) including:
    *   **Read Fidelity:** Comparing top-of-stack bits vs. ground truth to isolate logic errors from memory errors.
    *   **State Trajectories:** Monitoring 1D state "ramps" to identify OOD saturation.
    *   **Stack Distributions:** High-resolution heatmaps of stack activations.
3.  **The "Hard Action" Breakthrough:** Discovered that forcing discrete PUSH/POP actions during evaluation ("Hard Inference") allows the model to generalize to $L=140+$ despite training only on $L \le 60$.
4.  **Baseline Benchmark:** Confirmed that the "Soft Stack" baseline (purely differentiable) generalizes to $L=100$ after 10,000 steps but fails at $L=200$ due to State Component Drift.

## Current Project State

*   **Architecture:** `StackRNN` with learned embeddings and a 2D state controller.
*   **Performance:** 100% Sequence Accuracy up to $L=140$ (Hard Inference) and $L=100$ (Soft Inference).
*   **Infrastructure:** Support for high-depth stacks ($D=600$) and long-range OOD testing ($L=500$).

## Immediate Research Goals

### 1. State Size vs. Generalization
Investigate whether increasing the controller state size (currently 2D) leads to **length-coupled solutions**. 
*   **Hypothesis:** Higher-dimensional states provide more "capacity" to overfit specific training lengths, potentially degrading OOD accuracy compared to the minimal 2D "counter" representation.

### 2. Training Loss vs. Confidence Correlation
Correlate the training loss magnitude with OOD generalization robustness.
*   **Experiment:** Perform the same seeded run and evaluate the model at checkpoints (e.g., step 6,000, 8,000, 10,000).
*   **Analysis:** Determine if training past 100% ID accuracy makes the model "more confident" (less blurry stack actions) and if this directly translates to better OOD performance.

### 3. Task Expansion
Identify and implement two new algorithmic tasks that fit the current stack-based architecture:
*   **Dyck-1 (Parenthesis Matching):** Testing recursive PUSH/POP logic without a monotonic ramp.
*   **Palindrome Detection/XOR-Reversal:** Adding complexity to the bit-processing logic.

## Building and Running

### 1. Dependencies
`jax`, `flax`, `optax`, `matplotlib`, `seaborn`, `numpy`, `tqdm`.

### 2. Running Experiments
The main script is `run_experiment.py`. It trains a model (configured in `constants.py`) and generates a full suite of OOD plots in a timestamped or named folder in `results/`.

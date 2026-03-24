# Gemini Assistant Project Memory: StackRNN

This document provides a comprehensive overview of the StackRNN project to serve as a persistent context for AI-assisted development sessions.

## Project Overview

This is a Python-based research project investigating the out-of-distribution (OOD) generalization capabilities of memory-augmented Recurrent Neural Networks (RNNs). We are comparing how different architectures learn algorithmic tasks and generalize to inputs longer than those seen during training.

*   **Core Tasks:**
    *   **Bit-string Reversal:** Encoder-Decoder architecture for reversing sequences.
    *   **Dyck-1 (Parenthesis Matching):** Auto-regressive prefix completion task.
*   **Technologies:** JAX, Flax, Optax, Matplotlib, Seaborn.
*   **Key Insight:** Soft-stack training with **Hard Stack Inference** (discrete actions at test time) allows perfect algorithmic generalization by eliminating cumulative pointer drift.

## Development Log & Recent Breakthroughs

1.  **Codebase Reorganization:** Separated the project into task-specific modules (`reversal/` and `dyck1/`) with shared logic in `stack_utils.py` at the root.
2.  **Dyck-1 Implementation:** Successfully implemented the Dyck-1 task with:
    *   **OOD Generalization Plots:** Comparative study between Hard vs Soft stack inference across sequence lengths (up to $L=500$).
    *   **Refined Visualizations:** High-resolution heatmaps of stack probability distributions (P(Open)), growing upwards to reflect stack growth, with token annotations and no background grids.
3.  **State Dimensionality Study (Dyck-1):**
    *   **8D State:** Converges extremely fast (100% ID accuracy by step 3,000) but shows poor OOD generalization, potentially due to overfitting.
    *   **2D State:** Slower convergence but more likely to learn the minimal "counter" logic required for Dyck languages.
4.  **The "Hard Action" Breakthrough (Reversal):** Confirmed that forcing discrete PUSH/POP actions during evaluation allows generalization to $L=500+$ despite training only on $L \le 60$.

## Current Project State

*   **Structure:**
    *   `/reversal/`: String reversal task files.
    *   `/dyck1/`: Dyck-1 task files.
    *   `/results/`: Organized by task and experiment configuration.
*   **Performance:**
    *   **Reversal:** 100% Accuracy up to $L=500$ (Hard Inference).
    *   **Dyck-1:** 100% ID Accuracy; OOD generalization varies by state size (currently investigating).
*   **Infrastructure:** Support for high-depth stacks ($D=600$) and long-range OOD testing ($L=500$).

## Immediate Research Goals

### 1. String Reversal Task ($Q = \text{NUM\_STATES}$)
*   **$Q=64$ Hyperparameter Search:** Conduct a systematic search to ensure the model consistently reaches 100% training accuracy. This is critical to ensure that any OOD generalization failures are due to the learned solution's structure rather than simple underfitting.
*   **$Q=2$ Phase Change Analysis:** Investigate the empirical "plateau" behavior where the model lingers at 50% (random) accuracy for many steps before a sudden phase change to 100%. 
    *   **Analysis:** Determine if the bottleneck is learning the state transitions or the stack actions.
    *   **Method:** Monitor gradient norms and specific weights in the state-controller vs. memory-action heads during the plateau.

### 2. Dyck-1 Task
*   **Data Volume Study:** Investigate if the poor OOD generalization observed in Dyck-1 is simply a function of training data scarcity for a more complex class.
*   **Sample Size Scaling Experiment:**
    *   Define a range of training set sizes (number of unique samples).
    *   For each size, train the model to 100% ID accuracy.
    *   **Evaluation:** Generate OOD generalization curves (Accuracy vs. Test Sequence Length) for each training size to identify the threshold for reliable algorithmic learning.

### 3. Task Expansion
*   **Palindrome Detection:** Implementation of bit-string palindrome detection.
*   **Dyck-2:** Expanding to two bracket types to test nested recursive logic.

## Building and Running

### 1. Environment
Use the `jax_env` conda environment:
```bash
/opt/homebrew/anaconda3/envs/jax_env/bin/python
```

### 2. Running Experiments
Run the task-specific `run_experiment.py` within its directory. Results are saved to `../results/<task>/<exp_name>`.

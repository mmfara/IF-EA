# IF-EA: Integrating Fairness-Enhancing Techniques in AI Models for Recidivism Prediction

## Overview
This repository contains the Python code used to conduct the experiments described in the research paper:

**"Towards Trustworthy AI: Potential and Peril of Integrating Multi-Phase Bias Mitigation Techniques in Recidivism Models"**

[Michael Mayowa Farayola](https://github.com/mmfara), Irina Tal, Takfarinas Saber, Regina Connolly, Malika Bendechache.

The research explores the impact of integrating fairness-enhancing techniques across multiple phases (pre-processing, in-processing, and post-processing) in AI models for predicting recidivism risk. The study employs **multi-objective optimization** to identify fairness-accuracy trade-offs in recidivism prediction models.

## Repository Structure
```
IF-EA/
│── data/                 # Dataset used for experiments (COMPAS dataset processed)
│── models/               # Trained models and saved outputs
│── notebooks/            # Jupyter notebooks for running and visualizing experiments
│── scripts/              # Python scripts for executing bias mitigation techniques
│── results/              # Experimental results and analysis
│── requirements.txt      # Dependencies
│── README.md             # This file
│── main.py               # Main script for running experiments
```

## Installation and Setup
### Prerequisites
Ensure you have Python (>=3.8) installed. The required dependencies can be installed using:
```bash
pip install -r requirements.txt
```

### Dependencies
This project uses the following key Python libraries:
- `aif360` (IBM AI Fairness 360 toolkit)
- `scikit-learn` (Machine learning models and metrics)
- `numpy`, `pandas` (Data processing)
- `matplotlib`, `seaborn` (Visualization)

## Key Findings from the Research
- **Integrating fairness-enhancing techniques** across multiple phases improves fairness more effectively than applying them in isolation.
- Some multi-phase integrations maintain high predictive accuracy while **reducing bias**.
- Utilize Multi-objective optimization (MOO) to identify **Pareto-optimal models** that balance fairness and accuracy.
- Techniques such as integrated technique of **Disparate Impact Remover, Adversarial Learning, and Equalized Odds Optimization** show promising results in mitigating bias.

## Fairness Interventions Considered
The following fairness-enhancing techniques were explored in the study:

### Pre-processing Techniques
- **Reweighing (RW):** Adjusts instance weights in the training dataset to balance the distribution of privileged and unprivileged groups.
- **Disparate Impact Remover (DIR):** Modifies feature values in the dataset to reduce bias while preserving rank order.

### In-processing Techniques
- **Exponentiated Gradient Reduction (EGR):** Produces a randomized classifier that minimizes empirical error while enforcing fairness constraints.
- **Adversarial Learning (AL):** Introduces an adversary that attempts to predict protected attributes, encouraging the model to become fairer.

### Post-processing Techniques
- **Reject Option-Based Classification (ROC):** Adjusts classification decisions near the decision boundary to improve fairness.
- **Equalized Odds Optimization (EO):** Modifies predicted labels to ensure fairness across both true positive and false positive rates.

## Integrated Approach carried out in the Research
**Train the AI models with fairness interventions**
   - `baseline` (No fairness intervention)
   - `PI` (Pre-processing + In-processing techniques)
   - `PP` (Pre-processing + Post-processing techniques)
   - `IP` (In-processing + Post-processing techniques)
   - `PIP` (All three phases combined)

**Evaluate fairness metrics**
   This script computes fairness metrics including:
   - Statistical Parity Difference (SPD)
   - Disparate Impact (DI)
   - Equal Opportunity Difference (EOD)
   - Predictive Equality Difference (PED)

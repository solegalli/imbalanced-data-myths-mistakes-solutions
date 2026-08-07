[![License https://github.com/solegalli/imbalanced-data-myths-mistakes-solutions/blob/main/LICENSE](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/solegalli/imbalanced-data-myths-mistakes-solutions/blob/main/LICENSE)
[![Sponsorship https://www.trainindata.com/](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/)

## Imbalanced Data: Myths, Mistakes and Modern Solutions - Code Repository

- Published: August, 2026

[<img src="./MOCKUP_BOOK.jpg" width="248">](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

## Links

- [Book](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

## Table of Contents

- **1. Imbalanced Data: Class Frequency is Not The Problem**
  - Imbalanced Datasets: What Are They?
  - What Factors Influence the Classification of Imbalanced Datasets?
  - Resampling in The Age of GBMs
  - Prediction is Not Classification
  - How to Approach Imbalanced Learning
  - Myths, Mistakes and Modern Solutions
  - References

- **2. Metrics that Matter (and Pitfalls to Avoid)**
  - Understanding the Output of Machine Learning Models
  - Understanding What Metrics Measure
  - Threshold Dependent vs Threshold Independent Metrics
  - What Happens When We Threshold Predictions
  - Why Choosing The Right Metric Matters
  - How To Choose the Right Metric
  - Making Sense of Classification Metrics
  - Choosing the Right Threshold For Classification Metrics
  - Understanding Ranking Metrics
  - Myths, Mistakes and Modern Solutions
  - References

- **3. Probability Calibration: When 70% Means 70%**
  - Reliable Probability Estimates: What Are They?
  - Calibrated Probabilities: Why Do They Matter?
  - Assessing Probability Calibration: Reliability Diagrams
  - What Makes Calibration Assessment Hard
  - What Breaks Probability Calibration
  - Scoring Functions: Training Models to Be Calibrated
  - Recalibration: Correcting Biased Probabilities
  - Recalibrating Models in Python
  - Myths, Mistakes and Modern Solutions
  - References

- **4. Cost-Sensitive Learning: Thresholds, Weights, and Decisions**
  - Cost-sensitive Learning: What is it?
  - Making Cost-Sensitive Decisions
  - Thresholding, Weights, and Resampling Are Equivalent
  - Reweighting, Resampling, Thresholding: Different Paths to the Same Goal
  - Thresholds and Weights: Theory Meets Evidence
  - Empirical Thresholding: Finding the Right Decision Rule
  - When Cost is Not Class Frequency
  - Myths, Mistakes and Modern Solutions
  - References

- **5. Undersampling and Cleaning Methods: Discarding Hard-Won Data**
  - Undersampling: What Is It?
  - The Need for Undersampling: A Historical Perspective
  - The Value of Undersampling: A Mixed and Fragile Picture
  - A Modern Reassessment of Undersampling
  - Resampling is Threshold Shifting
  - Undersampling in Python
  - Myths, Mistakes and Modern Solutions
  - References

- **6. Oversampling and SMOTE: The Illusion of Better Models**
  - Oversampling: What Is It?
  - The Birth of Oversampling and SMOTE
  - The Rise of Oversampling and SMOTE
  - The Fall of Oversampling and SMOTE
  - A Modern Reassessment of Oversampling
  - Is There Any Place for Oversampling?
  - Myths, Mistakes and Modern Solutions
  - References

- **7. Imbalanced Learning in the Age of AI**
  - The Workflow We Want the Machine to Follow
  - Ask AI to Analyse the Data
  - Ask AI to Build the Model
  - Audit the Pipeline
  - Ask AI to Rebuild the Pipeline
  - How I Work With AI
  - Last Words on Imbalanced Data


## Buy the Book

- [Book](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

## Setup

If you want to run the recipes of this book in a dedicated environment:

**Create and activate a virtual environment**
```bash
python -m venv mlidbook
source mlidbook/bin/activate        # macOS/Linux
mlidbook\Scripts\activate           # Windows
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Install Jupyter and register the kernel**
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=mlidbook --display-name "mlidbook"
```

The environment will now be available as a kernel named **mlidbook** in Jupyter Notebook.
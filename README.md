# Disease Prediction using Bayesian Networks 🧠

A machine learning application that predicts diseases based on symptoms using Bayesian Networks. The model achieves 100% accuracy on both training and testing datasets.

## Model Overview

- **Diseases**: 41 different conditions
- **Symptoms**: 131 distinct symptoms
- **Dataset Size**: ~5,000 patient records
- **Accuracy**: 100% on training and testing sets
- **Technology**: Bayesian Network implementation using `bnlearn`

## Features

- Wordclouds of symptoms for each disease
- Interactive web interface built with Streamlit
- Real-time disease prediction
- Probability distribution for possible diseases
- Simple symptom selection interface with three states:
  - ✅ Present
  - ❌ Absent
  - ⬜ Uncertain

## Project Structure

- `app.py`: Main Streamlit application
- `bn_workflow.ipynb`: Jupyter notebook for data processing, visualization, and model training/evaluation
- `data/`: Dataset files
- `models/`: Saved model files
- `utils/`
  - `model.py`: Bayesian Network model operations
  - `visualization.py`: Visualization utilities
- `visualizations/wordclouds`: Wordclouds of symptoms for each disease

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bruno-rda/bn-disease-prediction.git
cd bn-disease-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Select symptoms using the interactive interface

4. Click "🔮 Predict Disease" to get predictions
# Credit Risk Analysis Dashboard

A full-stack machine learning application that predicts the likelihood of loan default based on applicant information. Users can input applicant details through a dashboard and receive a real-time default risk prediction.

## Overview

This project uses the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset to train an XGBoost model that predicts whether an applicant will default on a loan. The model is served via a Python microservice, orchestrated by a Node.js/Express backend, and presented through a React dashboard.

## Tech Stack

- **Frontend:** React
- **Backend:** Node.js, Express
- **ML Model Service:** Python, XGBoost
- **Dataset:** Home Credit Default Risk (Kaggle)

## Architecture

```
React Dashboard  -->  Express API  -->  Python ML Service  -->  XGBoost Model
     (input form)      (orchestration)     (inference)
```

The Express backend handles requests from the frontend, validates/formats input, and forwards it to a Python microservice responsible for running predictions against the trained XGBoost model. This separates concerns cleanly: Node handles web/API orchestration, Python handles ML inference.

## Project Status

🚧 In development

- [ ] Train baseline XGBoost model on `application_train.csv`
- [ ] Build Python prediction microservice
- [ ] Build Express API layer
- [ ] Build React input dashboard
- [ ] Connect full pipeline end-to-end
- [ ] Deploy

## Getting Started

### Prerequisites
- Node.js (vX.X)
- Python 3.x
- pip / virtualenv

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd credit-risk-predictor

# Install backend dependencies
cd backend
npm install

# Install frontend dependencies
cd ../frontend
npm install

# Install Python service dependencies
cd ../ml-service
pip install -r requirements.txt
```

### Running Locally

```bash
# Start the Python ML service
cd ml-service
python app.py

# Start the Express backend
cd backend
npm run dev

# Start the React frontend
cd frontend
npm start
```

## Dataset

This project uses `application_train.csv` and `application_test.csv` from the Home Credit Default Risk dataset, along with the accompanying column description file. Supplementary tables (bureau history, previous applications, etc.) may be incorporated in a later iteration for additional feature engineering.

## Roadmap / Future Improvements

- SHAP-based explainability for individual predictions
- Feature engineering using supplementary bureau/previous application data
- Prediction history storage (database)
- Authentication
- Cloud deployment (AWS)
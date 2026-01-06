# MLOps – Heart Disease Prediction

## Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for heart disease prediction.  
It covers data acquisition, exploratory data analysis (EDA), feature engineering, model development, CI/CD, experiment tracking, containerization, API deployment, monitoring, and documentation.

---

##  1. Setup & Installation

### Clone the Repository
```bash
git clone https://github.com/Sreedhar30/MLOps-Heart-Disease
cd MLOps-Heart-Disease
Create and Activate Python Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate         # Windows
# or
source venv/bin/activate      # Linux / Mac
Install Dependencies
bash
Copy code
pip install -r requirements.txt
2. Data and Exploratory Data Analysis (EDA)
Dataset: Heart disease CSV

Handled missing values with median/mode imputation

Scaled numeric features using StandardScaler

Encoded categorical features using OneHotEncoder

Visualizations included:
✔ Histograms
✔ Correlation heatmap
✔ Target distribution
✔ Missing values chart

(Screenshots in screenshots/EDA/)

3. Feature Engineering & Model Development
Models trained:

Logistic Regression

Random Forest Classifier

Best metrics logged via MLflow:
✔ Accuracy
✔ Precision
✔ Recall
✔ ROC-AUC

Random Forest selected as final model.
(Screenshots in screenshots/step2_model_metrics.png)

4. Experiment Tracking (MLflow)
Tracked:

Parameters

Metrics

Model artifacts

MLflow UI shows both model runs.
(Screenshots in screenshots/step3_Git.png)

5. CI/CD Pipeline
Automated testing using PyTest.
Linting using Flake8.
GitHub Actions workflow includes:
✔ Linting
✔ Unit testing
✔ Training execution

Tests located in tests/.
(Screenshots in screenshots/Step5/pytest sucess.png)

6. Model Packaging & API
Used FastAPI to serve model predictions.
Endpoints:

bash
Copy code
GET  /
POST /predict
GET  /metrics
Returns:
✔ Prediction
✔ Confidence
✔ API metrics

Containerized using Docker.
(Screenshots in screenshots/Step6)

7. Production Deployment
Deployed on local Kubernetes (Docker Desktop):

✔ Deployment manifest
✔ Service NodePort (http://localhost:30007/docs)

(Screenshots in screenshots/Step7/)

8. Monitoring & Logging
Implemented API request logging:
✔ Request path
✔ Status code
✔ Response time
✔ Total requests

Added metrics endpoint:

bash
Copy code
GET /metrics
(Screenshots in screenshots/Step8/)

9. Architecture Diagram
markdown
Copy code
User  →  FastAPI  →  Model Pipeline  →  Prediction
      ↕
      Logging  &  Metrics
      ↕
   Docker + Kubernetes
      ↕
  CI/CD (GitHub Actions)
(You can also include an image screenshots/architecture.png)

Repository
GitHub: https://github.com/Sreedhar30/MLOps-Heart-Disease

Folder Structure
css
Copy code
MLOps-Heart-Disease/
├── api/  
├── data/  
├── model/  
├── notebooks/  
├── screenshots/  
├── src/  
├── tests/  
├── Dockerfile.txt  
├── k8s/  
├── requirements.txt  
└── README.md
🔚 Conclusion
This project demonstrates a full MLOps lifecycle:
✔ Data engineering
✔ Model training
✔ Experiment tracking
✔ CI/CD
✔ Containerization
✔ Deployment
✔ Logging & Monitoring
✔ Documentation


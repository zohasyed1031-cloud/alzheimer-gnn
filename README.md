\# 🧠 Alzheimer’s Disease Detection Using Gene Expression and Graph Neural Networks



An interactive web-based research prototype for analyzing gene-expression data and generating \*\*AD / MCI / CTL model predictions\*\* using a Graph Neural Network (GNN).



The system combines a \*\*FastAPI backend\*\*, \*\*React frontend\*\*, and a trained \*\*Graph Convolutional Network (GCN)\*\* to provide an interactive dashboard for gene-expression analysis and prediction.



> \*\*Important:\*\* This project is a research/academic prototype and is \*\*not a clinically validated diagnostic system\*\*. Model predictions should not be used for medical diagnosis or treatment decisions.



\---



\## 📌 Project Overview



Alzheimer’s disease is a progressive neurodegenerative disorder that can be difficult to identify during its early stages.



This project explores the use of \*\*gene-expression data and Graph Neural Networks\*\* to classify samples into three categories:



\* \*\*AD\*\* — Alzheimer’s Disease

\* \*\*MCI\*\* — Mild Cognitive Impairment

\* \*\*CTL\*\* — Control / Healthy



The application provides an interactive interface where users can upload a gene-expression CSV file, analyze the available features, and obtain a prediction from the trained GNN model.



\---



\## ✨ Features



\### 🧬 Gene Expression Analysis



\* Upload gene-expression CSV files

\* Extract required gene features

\* Analyze gene-expression characteristics

\* Display selected/high-variance genes

\* Visualize gene-expression patterns interactively



\### 🧠 GNN-Based Prediction



The application uses a trained Graph Convolutional Network to classify gene-expression samples into:



| Class  | Meaning                   |

| ------ | ------------------------- |

| 🔴 AD  | Alzheimer’s Disease       |

| 🟠 MCI | Mild Cognitive Impairment |

| 🟢 CTL | Control / Healthy         |



The model uses \*\*1,000 selected genes\*\* as graph nodes.



\### 📊 Interactive Dashboard



The frontend provides:



\* Dataset analysis

\* Gene statistics

\* Interactive charts

\* GNN model information

\* Prediction probabilities

\* Prediction visualization

\* Patient report generation

\* Blood sample selection



\---



\## 🏗️ System Architecture



```text

&#x20;                ┌──────────────────────┐

&#x20;                │   React Frontend     │

&#x20;                │   Interactive UI     │

&#x20;                └──────────┬───────────┘

&#x20;                           │

&#x20;                           │ HTTP / REST API

&#x20;                           ▼

&#x20;                ┌──────────────────────┐

&#x20;                │   FastAPI Backend    │

&#x20;                │                      │

&#x20;                │ Data Processing      │

&#x20;                │ Feature Extraction   │

&#x20;                │ GNN Prediction       │

&#x20;                └──────────┬───────────┘

&#x20;                           │

&#x20;                           ▼

&#x20;                ┌──────────────────────┐

&#x20;                │  Trained GNN Model   │

&#x20;                │                      │

&#x20;                │ 1000 Gene Features   │

&#x20;                │ k-NN Graph           │

&#x20;                │ Weighted GCN         │

&#x20;                └──────────────────────┘

```



\---



\## 🔬 Machine Learning Pipeline



The project follows this workflow:



```text

Gene Expression Dataset

&#x20;         ↓

Data Cleaning

&#x20;         ↓

Gene Annotation

&#x20;         ↓

Probe-to-Gene Mapping

&#x20;         ↓

Duplicate Gene Handling

&#x20;         ↓

Quality Validation

&#x20;         ↓

Variance Filtering

&#x20;         ↓

Feature Selection

&#x20;         ↓

Top 1000 Genes

&#x20;         ↓

k-NN Graph Construction

&#x20;         ↓

Feature Standardization

&#x20;         ↓

Weighted GCN

&#x20;         ↓

AD / MCI / CTL Prediction

&#x20;         ↓

Prediction Probabilities

&#x20;         ↓

Interactive Patient Report

```



\---



\## 🧬 Dataset



The primary dataset used during model development is:



\*\*GSE63060\*\*



The dataset contains blood gene-expression samples associated with:



\* Alzheimer’s Disease (AD)

\* Mild Cognitive Impairment (MCI)

\* Control (CTL)



\### Dataset Summary



| Property        | Value |

| --------------- | ----: |

| Total samples   |   329 |

| AD samples      |   145 |

| MCI samples     |    80 |

| Control samples |   104 |

| Selected genes  | 1,000 |

| Graph type      |  k-NN |

| k value         |    10 |



The original raw and processed datasets are \*\*not included in this repository\*\* because of their size and dataset distribution considerations.



\---



\## 🧠 Graph Neural Network



The project represents the selected genes as nodes in a graph.



A \*\*k-nearest-neighbor (k-NN)\*\* graph is constructed to represent relationships between genes.



\### Graph Configuration



```text

Number of nodes: 1000

k-NN value: 10

Graph edges: 9141 undirected edges

Edge index: 18282 directed entries

```



The trained model is a weighted Graph Convolutional Network.



\### Model Architecture



```text

Input

1000 gene nodes

&#x20;      ↓

GCN Layer

1 → 64

&#x20;      ↓

ReLU

&#x20;      ↓

GCN Layer

64 → 64

&#x20;      ↓

ReLU

&#x20;      ↓

Dropout

&#x20;      ↓

Global Mean Pooling

&#x20;      +

Global Max Pooling

&#x20;      ↓

128 features

&#x20;      ↓

Fully Connected Layer

128 → 3

&#x20;      ↓

AD / CTL / MCI

```



The trained model and supporting artifacts are stored in:



```text

backend/model/

```



\---



\## 🛠️ Technology Stack



\### Frontend



\* React

\* Vite

\* JavaScript

\* CSS

\* Recharts

\* Lucide React



\### Backend



\* Python

\* FastAPI

\* Uvicorn

\* Pandas

\* NumPy

\* PyTorch

\* PyTorch Geometric

\* Joblib



\### Machine Learning



\* Graph Convolutional Network (GCN)

\* k-NN graph construction

\* Feature selection

\* Feature standardization

\* Weighted classification



\---



\## 📂 Project Structure



```text

alzheimer-gnn/

│

├── backend/

│   ├── main.py

│   └── model/

│       ├── edge\_index.pt

│       ├── edge\_weight.pt

│       ├── scaler.pkl

│       ├── selected\_genes\_1000.csv

│       └── weighted\_improved\_gcn.pth

│

├── frontend/

│   ├── public/

│   ├── src/

│   │   ├── App.jsx

│   │   ├── App.css

│   │   ├── index.css

│   │   └── main.jsx

│   ├── package.json

│   ├── package-lock.json

│   └── vite.config.js

│

├── notebooks/

│   └── 01\_dataset\_download...

│

├── .gitignore

└── README.md

```



\---



\# 🚀 Installation



\## 1. Clone the Repository



```bash

git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd alzheimer-gnn

```



\---



\## 2. Backend Setup



Create a Python virtual environment:



\### Windows



```powershell

python -m venv venv

```



Activate it:



```powershell

venv\\Scripts\\activate

```



Install the required dependencies:



```powershell

pip install fastapi uvicorn pandas numpy joblib torch torch-geometric

```



> PyTorch and PyTorch Geometric installation requirements can vary depending on the operating system and CPU/GPU configuration.



\---



\## 3. Start the FastAPI Backend



From the project root:



```powershell

cd backend

uvicorn main:app --reload

```



The backend will run at:



```text

http://127.0.0.1:8000

```



You can verify the backend by opening:



```text

http://127.0.0.1:8000/

```



\---



\# 💻 Frontend Setup



Open another terminal.



From the project root:



```powershell

cd frontend

npm install

```



Start the React development server:



```powershell

npm run dev

```



If PowerShell blocks `npm.ps1`, use:



```powershell

npm.cmd run dev

```



Vite will provide a local address such as:



```text

http://localhost:5173

```



Open that address in your browser.



\---



\# 📄 Patient CSV Format



For GNN prediction, the uploaded patient/sample CSV must contain the \*\*1,000 gene features used by the trained model\*\*.



Example structure:



```text

Patient\_ID,Sample\_Type,Diagnosis,AB42,AB40,pTau181,pTau217,AB42\_AB40\_Ratio,ABCA7,ABCE1,ACADM,...

```



The exact gene columns required by the model are available in:



```text

backend/model/selected\_genes\_1000.csv

```



\### Important



The current GNN prediction pipeline requires all 1,000 selected gene features.



A CSV containing only a few genes cannot be directly passed to the trained model because the model was trained using the complete 1,000-gene feature space.



\---



\# 📊 Prediction Output



The application generates probabilities for the three model classes:



```text

AD

MCI

CTL

```



The dashboard visualizes these probabilities interactively.



The probability visualization is a \*\*model output\*\*, not a medically validated clinical risk score.



\---



\# 🔐 Data and Privacy



Do not upload:



\* Personal medical information

\* Personally identifiable patient information

\* Private API keys

\* Passwords

\* `.env` files

\* Sensitive clinical records



The repository intentionally excludes:



```text

venv/

node\_modules/

data/raw/

data/processed/

patient\_gnn\_demo.csv

```



\---



\# ⚠️ Research Disclaimer



This software is developed for \*\*academic and research purposes\*\*.



The model has not been clinically validated and should not be considered a medical diagnostic tool.



Predictions generated by the system are machine-learning outputs and should not replace professional medical evaluation, laboratory testing, or clinical diagnosis.



\---



\# 🎯 Project Goals



The project aims to demonstrate how Graph Neural Networks can be applied to high-dimensional gene-expression data for Alzheimer's disease research.



Future development may include:



\* Blood and saliva sample support

\* Additional biomarkers

\* Larger multi-dataset validation

\* Improved graph construction

\* Explainable AI

\* Gene importance visualization

\* External dataset validation

\* Improved model generalization

\* Clinical validation research



\---



\# 👩‍💻 Project



\*\*Alzheimer’s Disease Detection Using Gene Expression and Graph Neural Networks\*\*



Developed as an academic major project exploring:



\*\*Bioinformatics + Machine Learning + Graph Neural Networks + Interactive Web Applications\*\*



\---




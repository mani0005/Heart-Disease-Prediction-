# Heart Disease Prediction using Machine Learning

This project is an **end-to-end heart disease prediction system** built using **Python, Scikit-learn, and Flask**.  
It uses real-world health data to train machine learning models and allows users to input their details through a web interface to check their heart disease risk.

---

## Features

- Data Cleaning & Preprocessing (handling missing values, removing unnecessary columns)
- Data Balancing using **SMOTE + RandomUnderSampler**
- Feature Scaling with **StandardScaler**
- Model Training using:
  - **K-Nearest Neighbors (KNN)**
  - **Logistic Regression**
- Model Evaluation with accuracy and confusion matrix
- Model Saving with `joblib` (`.pkl` files)
- Flask Web Application for user input and real-time predictions
- Confidence Score and personalized prediction message

---

## 🗂️ Project Structure

heart-disease-prediction/
│
├── app.py                        # Flask web application (runs the web UI)
│
├── train_models.py               # Model training script (ML pipeline)
├── inspect_models.py             # Script to inspect saved .pkl models
│
├── dataset/
│   └── framingham.csv            # Original dataset
│
├── models/                       # Folder for saved ML models
│   ├── knn_model.pkl
│   ├── logreg_model.pkl
│   ├── scaler.pkl
│   └── features.json
│
├── templates/                    # HTML files for Flask frontend
│   └── index.html
│
├── static/                       # (optional) CSS, JS, or images
│   ├── style.css
│   └── logo.png
│
├── requirements.txt              # All dependencies
├── README.md                     # Project documentation
├── .gitignore                    # To ignore unnecessary files/folders
└── LICENSE                       # (optional) MIT or other open license



---

## How It Works

1. **Data Loading & Cleaning**  
   The dataset (`framingham.csv`) is read, unnecessary columns (like `education`) are dropped, and missing rows are removed.

2. **Feature Selection**  
   The top features used:
   ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

3. **Model Training**  
- Logistic Regression is trained for interpretability.
- KNN is trained for non-linear pattern recognition.
- Both models are calibrated and saved as `.pkl` files.

4. **Flask Web App**  
Users input their details → data is scaled → model predicts risk → result & confidence displayed.

---

#**Prerequisites**

## Installation of required software and Libraries (One Time)

1. Install the Anaconda Python Package
2. Open Anaconda Prompt and Move to the downloaded project directory (Heart Attack Risk Prediction) using the cd command

	Example:
	>> cd Path_of_Project_Directory
	
3. Create the virtual environment using the below command
	>>conda create -n harp python==3.11.7
4. Activate the virtual environment using the command
	>>conda activate harp
5. Now install the required Libraries using the below command
	>>pip install -r requirements.txt


## Steps to train the model after Installation of required software and Libraries

1. Open Anaconda Prompt and Move to the downloaded project directory (Heart Attack Risk Prediction) using the cd command

	Example:
	>> cd Path_of_Project_Directory
	
2. Activate the virtual environment using the command
	>>conda activate harp
	
	Note: harp is the environment created at the time of installing the software and Libraries
	
3. Next to train the model open the Jupyter Notebook using the below command
	>>jupyter notebook
4. Open the Heart-Attack-Risk-Prediction.ipynb and run all cells
5. Once the training is completed the trained model knn_model.pkl and logreg_model.pkl will be stored in the models directory


## Steps to run the Flask App after training the model 

1. Open Anaconda Prompt and Move to the downloaded project directory (Heart Attack Risk Prediction) using the cd command

	Example:
	>> cd Path_of_Project_Directory
	
2. Activate the virtual environment using the command
	>>conda activate harp
	
	Note: harp is the environment created at the time of installing the software and Libraries
	
3. Run the Flask App using the below command
	>>python app.py

## Tech Stack

1. **Language**: Python

2. **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn, Seaborn, Matplotlib

3. **Backend**: Flask

4. **Storage**: Joblib (for serialized models)

5. **Visualization**: Matplotlib, Seaborn



3. **Model Training**  
- Logistic Regression is trained for interpretability.
- KNN is trained for non-linear pattern recognition.
- Both models are calibrated and saved as `.pkl` files.

4. **Flask Web App**  
Users input their details → data is scaled → model predicts risk → result & confidence displayed.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mani005/heart-disease-prediction.git
cd heart-disease-prediction






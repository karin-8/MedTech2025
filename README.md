# 🏥 Applied Machine Learning in Healthcare — Course Notebooks

Welcome to the **Applied Machine Learning in Healthcare** notebook series.  
This course takes you from **Python basics** all the way to **AI engineering & deployment** in healthcare contexts.

---

## 📘 Course Overview

This repository contains four Jupyter/Colab notebooks:

1. **Notebook 1 — Introduction to Python in Healthcare**  
   - Setup Python environment  
   - Install and check ML libraries  
   - Run your first interactive calculation  

2. **Notebook 2 — Computer Vision in Healthcare**  
   - Preprocess chest X-ray images  
   - Apply **transfer learning (ResNet)**  
   - Train & evaluate a pneumonia detection model  
   - Reflect on AI ethics in diagnosis  

3. **Notebook 3 — Machine Learning with Tabular Data**  
   - Work with **structured clinical data** (Pima Indians Diabetes dataset)  
   - Explore, clean, and preprocess features  
   - Train classifiers (Random Forest, Logistic Regression, SVM)  
   - Evaluate with metrics, confusion matrix, ROC-AUC  

4. **Notebook 4 — AI Engineering**  
   - Deploy a model with **FastAPI** (conceptual demo)  
   - Track experiments with **MLflow**  
   - Understand **model monitoring** & **automatic retraining**  
   - Discuss healthcare deployment risks  

---

## 🚀 Getting Started

### Run in Google Colab (Recommended)
Each notebook is designed to run on Google Colab.  
- Open notebook link in Colab  
- Follow instructions inside the notebook  

### Local Setup
If you want to run locally:
```bash
git clone https://github.com/karin-8/MedTech2025.git
cd healthcare-ml-course
pip install -r requirements.txt
````

---

## 📂 Repository Structure

```
.
├── Notebook1_Intro_Python_Healthcare.ipynb
├── Notebook2_Computer_Vision_Healthcare.ipynb
├── Notebook3_Tabular_Data_Healthcare.ipynb
├── Notebook4_AI_Engineering.ipynb
├── README.md
└── requirements.txt
```

---

## 🧰 Requirements

* Python 3.9+
* [scikit-learn](https://scikit-learn.org/)
* [PyTorch](https://pytorch.org/)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [FastAPI](https://fastapi.tiangolo.com/) (for Notebook 4 demo)
* [MLflow](https://mlflow.org/) (for monitoring demo)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🌍 Ethical Disclaimer

These notebooks are for **educational purposes only**.
The datasets and models used here are **not intended for clinical use**.
Always consult qualified medical professionals for healthcare decisions.

---

## 🙌 Acknowledgements

* [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)
* [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) (referenced in Notebook 2 context)
* Open-source ML and healthcare research community

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

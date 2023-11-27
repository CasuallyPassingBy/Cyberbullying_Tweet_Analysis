# Cyberbullying Tweet Analysis Project
This project focuses on the analysis of cyberbullying tweets categorized by cyberbullying types. The dataset contains cleaned text, T-SNE plot coordinates, and categorization by cyberbullying types. The main goal is to predict cyberbullying types for new tweets using traditional Machine Learning Models and a Convolutional Neural Network (CNN).

## Dataset
The dataset includes:

**Cleaned Text:** Preprocessed text data.
- **T-SNE Plot Coordinates:** Coordinates for T-SNE visualization.
- **Cyberbullying Types:** Categorized types for each tweet.

Due to size constraints on GitHub (25 MB cap), the vectorized words cannot be included in the repository as it exceeds the limit even when compressed (85 MB).

## Accesing Necesary Files
Access the necessary files for the app's functionality via this [Google Drive link](https://drive.google.com/drive/folders/1jrwj2LWLXqOwQOdHwMlSJe9pMTHdTll1?usp=sharing):

- **Models:** Random Forest, Support Vector Machine, Logistic Regression, Convolutional Neural Network.
- **Scaler:** Necessary for model predictions.
- **Vectorized Texts:** Preprocessed and vectorized text data.

## Implemented Models
Implemented Machine Learning Models include:

- **Random Forest**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Convolutional Neural Network (CNN)**

## Functionality
The main Python file, project.py, utilizes Streamlit for data visualization and machine learning predictions. It offers the following functionalities:
- **Visualization:** Showcases word clouds, bar charts displaying the top 15 most common n-grams based on selected cyberbullying types, and a T-SNE plot for data visualization.
- **Machine Learning Predictions:** Given a tweet, the models generate probabilities for each cyberbullying category. Users can view these probabilities for different categories.

## Usage
To run the project locally:

1. Ensure Python and necessary libraries are installed.
2. Clone the repository.
3. Access the models, scaler, and vectorized texts from the provided Google Drive link.
4. Place the files in the appropriate directory.
5. Modify the path variable to the appropiate directory
6. Execute project.py using Streamlit.

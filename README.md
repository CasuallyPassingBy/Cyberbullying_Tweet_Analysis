# Cyberbullying Tweet Analysis Project
This project focuses on the analysis of cyberbullying tweets categorized by cyberbullying types. The dataset contains cleaned text, T-SNE plot coordinates, and categorization by cyberbullying types. The main goal is to predict cyberbullying types for new tweets using traditional Machine Learning Models.

## Dataset
The dataset includes:

**Cleaned Text:** Preprocessed text data.
- **T-SNE Plot Coordinates:** Coordinates for T-SNE visualization.
- **Cyberbullying Types:** Categorized types for each tweet.

Due to size constraints on GitHub (25 MB cap), the vectorized words cannot be included in the repository as it exceeds the limit even when compressed (85 MB).

## Machine Learning Models
Implemented Machine Learning Models include:

- **Random Forest**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**

## Functionality
The main Python file, projecto.py, utilizes Streamlit for data visualization and machine learning predictions. It offers the following functionalities:
Visualization: Showcases word clouds, bar charts displaying the top 15 most common n-grams based on selected cyberbullying types, and a T-SNE plot for data visualization.
Machine Learning Predictions: Given a tweet, the models generate probabilities for each cyberbullying category. Users can view these probabilities for different categories.

## Usage
To run the project locally:
Ensure Python and necessary libraries are installed.
Clone the repository.
Execute project.py using Streamlit.

## IMDB Movie Sentiment Classifier

## Overview
This project trains a small neural network to classify the overall sentiment (positive or negative) of movie reviews from the IMDb Large Movie Review Dataset. It demonstrates:

- Data loading & normalization in Python  
- Building and compiling a Keras model  
- Saving the trained model
- Training with validation split  
- Evaluating model accuracy  
- Visualizing loss curves
- Testing the trained model on any number of user generated movie reviews using a separate script

## Structure
- **model.py**: Complete script for data prep, model creation, training, evaluation, and plotting.  Saves trained model to .h5 file.
- **predict_reviews.py** Script for preprocessing the user generated movie reviews and making sentiment predictions using the trained model.
- **loss_curves.png**: Sample plot showing training and validation loss over epochs.
- **requirements.txt**: List of required packages and dependencies.

## How to Run
1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt


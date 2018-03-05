from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np


def fit_reg(X, Y):
  """
  Fits a linear regression model to the data and returns regression model with score and predictions"""
  reg = LinearRegression()
  reg.fit(X, Y)
  preds = reg.predict(X)
  score = pearsonr(Y, preds)
  return reg, score, preds


def predict_mapping_outputs(features, weights):
  """
  Predict outputs given input features and weights."""
  model_pcs = np.matmul(features - weights['pca_b'], weights['pca_w'])
  preds = np.matmul(model_pcs, weights['pls_w']) + weights['pls_b']
  return preds


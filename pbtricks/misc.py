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


def sort_df_rows(df, column_name, sorted_list):
  """
  Sort values in a column give the sorted list values
  :param df: dataframe object
  :param column_name: string containing the column name
  :param sorted_list: a list containing the ordered list of values
  :return: dataframe object
  """
  tmp_df = df.copy(deep=True)
  sorterIndex = dict(zip(sorted_list, range(len(sorted_list))))
  tmp_df['sorter_list'] = tmp_df[column_name].map(sorterIndex)

  tmp_df.sort_values(['sorter_list'], inplace=True)
  tmp_df.drop('sorter_list', 1, inplace=True)
  _ = tmp_df.reset_index(drop=True)
  return tmp_df
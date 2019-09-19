# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:01:00 2018

@author: Sharada

Made several modifications
"""

import tensorflow as tf
import math
import numpy as np
import pdb

"""
Four to twelve mapping according to NPSS paper
"""

# Change to normalize over ALL values?
# If we are considering each of the MFSC features to be independent, then the version below is correct
def norm_data(x):
  """
  x has shape of (n, 60)
  """
  min_val = np.amin(x, axis=0)
  max_val = np.amax(x, axis=0)
  return (x - min_val) / (max_val - min_val)

def get_sigma(scale, skewness, gamma_s):
  sigma1 = scale * tf.exp((gamma_s * tf.abs(skewness) - 1) * 0)
  sigma2 = scale * tf.exp((gamma_s * tf.abs(skewness) - 1) * 1)
  sigma3 = scale * tf.exp((gamma_s * tf.abs(skewness) - 1) * 2)
  sigma4 = scale * tf.exp((gamma_s * tf.abs(skewness) - 1) * 3)
  return sigma1, sigma2, sigma3, sigma4

"""
Replaced by the one that we used before, 
where we replaced k by i on the summation term -- change given in comments
"""
def get_mu(location, sigma1, sigma2, sigma3, sigma4, skewness, gamma_u):
  mu1 = location   # Same in both cases
  mu2 = location + sigma1 * gamma_u * skewness
  mu3 = location + (sigma1 + sigma2) * gamma_u * skewness
  mu4 = location + (sigma1 + sigma2 + sigma3) * gamma_u * skewness
  return mu1, mu2, mu3, mu4

# Check this
def get_w(skewness, shape, gamma_w):
  # calculate denominater
  den = tf.zeros(tf.shape(skewness))
  for i in range(4):
    den += tf.pow(skewness, tf.constant(2.0 * i)) * \
            tf.pow(shape, tf.constant(i, dtype = tf.float32)) * \
            tf.pow(gamma_w, tf.constant(i, dtype = tf.float32))   # cast power term to float

  w1 = tf.div(tf.pow(skewness, tf.constant(2.0 * 0)) * tf.pow(shape, tf.constant(0.0)) * tf.pow(gamma_w, tf.constant(0.0)), den)
  w2 = tf.div(tf.pow(skewness, tf.constant(2.0 * 1)) * tf.pow(shape, tf.constant(1.0)) * tf.pow(gamma_w, tf.constant(1.0)), den)
  w3 = tf.div(tf.pow(skewness, tf.constant(2.0 * 2)) * tf.pow(shape, tf.constant(2.0)) * tf.pow(gamma_w, tf.constant(2.0)), den)
  w4 = tf.div(tf.pow(skewness, tf.constant(2.0 * 3)) * tf.pow(shape, tf.constant(3.0)) * tf.pow(gamma_w, tf.constant(3.0)), den)
  return w1, w2, w3, w4

def four_to_twelve_mapping(out_a0, out_a1, out_a2, out_a3, 
                            gamma_u = tf.constant(1.6), 
                            gamma_s = tf.constant(1.1), 
                            gamma_w = tf.constant(1/1.75)):
  """
  conversion from a0,a1,a2,a3 to mu0-mu3, sigma0-sigma3, w0-w3. All are tensors
  """
  # location, scale, skewness, shape are all shape of (1, n, 60)
  location = 2 * tf.sigmoid(out_a0) - 1
  scale =  (2.0/255) * tf.exp(4 * tf.sigmoid(out_a1))
  skewness = 2 * tf.sigmoid(out_a2) - 1 
  shape = 2 * tf.sigmoid(out_a3)

  sigma1, sigma2, sigma3, sigma4 = get_sigma(scale, skewness, gamma_s = gamma_s)
  mu1, mu2, mu3, mu4 = get_mu(location, sigma1, sigma2, sigma3, sigma4, skewness, gamma_u = gamma_u)
  w1, w2, w3, w4 = get_w(skewness, shape, gamma_w = gamma_w)

  return mu1, mu2, mu3, mu4, sigma1, sigma2, sigma3, sigma4, w1, w2, w3, w4

# Check the way this has been split
def get_mixture_coef(output):
  """
  output is the output of wavenet, has shape of (1, n, CMG_channels)
  return 12 matrix, each of them being a matrix of shape (1, n, CMG_channels)
  """
  out_a0, out_a1, out_a2, out_a3 = tf.split(output, num_or_size_splits=4, axis=-1)
  # print(out_a0.get_shape().as_list())

  return four_to_twelve_mapping(out_a0, out_a1, out_a2, out_a3)

def temp_control(mu1, mu2, mu3, mu4, 
                sigma1, sigma2, sigma3, sigma4, 
                w1, w2, w3, w4,
                tau):
  mu_bar = mu1 * w1 + mu2 * w2 + mu3 * w3 + mu4 * w4
  mu1_hat = mu1 + (mu_bar - mu1) * (1 - tau)
  mu2_hat = mu2 + (mu_bar - mu2) * (1 - tau)
  mu3_hat = mu3 + (mu_bar - mu3) * (1 - tau)
  mu4_hat = mu4 + (mu_bar - mu4) * (1 - tau)
  sigma1_hat = sigma1 * tf.sqrt(tau)
  sigma2_hat = sigma2 * tf.sqrt(tau)
  sigma3_hat = sigma3 * tf.sqrt(tau)
  sigma4_hat = sigma4 * tf.sqrt(tau)

  return mu1_hat, mu2_hat, mu3_hat, mu4_hat, sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat

# An explicitly defined NLL loss
def nll_loss(mu1_hat, mu2_hat, mu3_hat, mu4_hat, 
                sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat, 
                w1, w2, w3, w4, 
                x):
    eps = 1e-10 # To prevent nan
    logprob = tf.log(w1 + eps) + tf.divide(tf.square(x - mu1_hat), 2 * tf.square(sigma1_hat + eps)) + tf.log(tf.square(sigma1_hat) + eps) \
            + tf.log(w2 + eps) + tf.divide(tf.square(x - mu2_hat), 2 * tf.square(sigma2_hat + eps)) + tf.log(tf.square(sigma2_hat) + eps) \
            + tf.log(w3 + eps) + tf.divide(tf.square(x - mu3_hat), 2 * tf.square(sigma3_hat + eps)) + tf.log(tf.square(sigma3_hat) + eps) \
            + tf.log(w4 + eps) + tf.divide(tf.square(x - mu4_hat), 2 * tf.square(sigma4_hat + eps)) + tf.log(tf.square(sigma4_hat) + eps)
    # print(logprob.get_shape().as_list())
    nll = tf.reduce_sum(logprob, axis = -1)
    nll = tf.reduce_mean(nll, axis = -1)
    # nll = tf.reduce_mean(logprob, axis = 1)
    # nll = tf.reduce_sum(nll, axis = -1)
    
    return nll

# Older loss
def get_lossfunc(mu1_hat, mu2_hat, mu3_hat, mu4_hat, 
                sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat, 
                w1, w2, w3, w4, 
                y): 

  d1 = tf.distributions.Normal(loc = mu1_hat, scale = sigma1_hat)
  d2 = tf.distributions.Normal(loc = mu2_hat, scale = sigma2_hat)
  d3 = tf.distributions.Normal(loc = mu3_hat, scale = sigma3_hat)
  d4 = tf.distributions.Normal(loc = mu4_hat, scale = sigma4_hat)

  prob = w1 * d1.prob(y) + w2 * d2.prob(y) + w3 * d3.prob(y) + w4 * d4.prob(y)
  prob = prob + 1e-5
  logprob = -1.0 * tf.log(prob)
  result = tf.reduce_sum(logprob, axis = -1)
  return tf.reduce_mean(result, axis = -1)


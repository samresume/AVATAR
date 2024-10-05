"""AVATAR: Adversarial Autoencoders with Autoregressive Refinement for Time Series Generation.
"""

# Necessary Packages
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from metrics.discriminative_metrics import discriminative_score_metrics

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator
import math
# ------------------------------------------------------------------------------------


def avatar(ori_data, parameters, num_samples):
  """AVATAR function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: AVATAR network parameters
    - num_samples: It can be “same” or a number, such as 1000, representing the number of samples to be generated. If “same” is entered, the same amount of original data will be generated.
    
  Returns:
    - generated_data: generated time-series data
  """
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
  def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
              
  ## Build a RNN networks          
  
  # Network Parameters
  if parameters['hidden_dim'] == 'same':
    hidden_dim = dim
  elif isinstance(parameters['hidden_dim'], int):  # Check if it is a number
    hidden_dim = parameters['hidden_dim']
  else:
    raise ValueError("hidden_dim must be 'same' or a numeric value") 
    
  num_layers   = parameters['num_layer']
  iterations   = parameters['iterations']
  batch_size   = parameters['batch_size']
  module_name  = parameters['module'] 
  z_dim        = hidden_dim
    
  # Input place holders
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "myinput_z")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")

# ------------------------------------------------------------------------------------
  
  def embedder(X, T):
    """Embedding network between original feature space to latent space.

    Args:
      - X: input time-series features
      - T: input time information

    Returns:
      - H: embeddings
    """
    with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
        # Initial input is X
        inputs = X

        for layer_num in range(num_layers):
            # Step 1: Apply batch normalization to the inputs of the current layer
            # Since 'is_training' is removed, we'll set 'training=True' to ensure batch normalization uses batch statistics
            normalized_inputs = tf.compat.v1.layers.batch_normalization(inputs, training=True)
            
            # Step 2: Create a GRU cell
            gru_cell = tf.compat.v1.nn.rnn_cell.GRUCell(
                num_units=hidden_dim,
                activation=tf.nn.tanh
            )
            
            # Step 3: Build the RNN for the current layer using the normalized inputs
            outputs, states = tf.compat.v1.nn.dynamic_rnn(
                gru_cell,
                normalized_inputs,
                dtype=tf.float32,
                sequence_length=T
            )
            
            # The outputs become the inputs for the next layer
            inputs = outputs

        # After the last layer, apply a dense layer with sigmoid activation
        H = tf.compat.v1.layers.dense(inputs, hidden_dim, activation=tf.nn.sigmoid)

    return H
      
  def recovery(H, T):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """
    with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
        # Initial input is H
        inputs = H

        for layer_num in range(num_layers):
            # Step 1: Apply batch normalization to the inputs of the current layer
            normalized_inputs = tf.compat.v1.layers.batch_normalization(inputs, training=True)

            # Step 2: Create a GRU cell
            gru_cell = tf.compat.v1.nn.rnn_cell.GRUCell(
                num_units=hidden_dim,
                activation=tf.nn.tanh
            )

            # Step 3: Build the RNN for the current layer using the normalized inputs
            outputs, states = tf.compat.v1.nn.dynamic_rnn(
                gru_cell,
                normalized_inputs,
                dtype=tf.float32,
                sequence_length=T
            )

            # The outputs become the inputs for the next layer
            inputs = outputs

        # After the last layer, apply a dense layer with sigmoid activation
        X_tilde = tf.compat.v1.layers.dense(inputs, dim, activation=tf.nn.sigmoid)

    return X_tilde
    
      
  def supervisor(X, T):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """
    with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
        # Initial input is H
        inputs = X

        for layer_num in range(num_layers - 1):
            # Step 1: Apply batch normalization to the inputs of the current layer
            normalized_inputs = tf.compat.v1.layers.batch_normalization(inputs, training=True)

            # Step 2: Create a GRU cell
            gru_cell = tf.compat.v1.nn.rnn_cell.GRUCell(
                num_units=hidden_dim,
                activation=tf.nn.tanh
            )

            # Step 3: Build the RNN for the current layer using the normalized inputs
            outputs, states = tf.compat.v1.nn.dynamic_rnn(
                gru_cell,
                normalized_inputs,
                dtype=tf.float32,
                sequence_length=T
            )

            # The outputs become the inputs for the next layer
            inputs = outputs

        # After the last layer, apply a dense layer with sigmoid activation
        S = tf.compat.v1.layers.dense(inputs, hidden_dim, activation=tf.nn.sigmoid)

    return S
          
  def discriminator (H, T):
    """Discriminate the original and synthetic time-series data.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """        
    with tf.compat.v1.variable_scope("discriminator", reuse = tf.compat.v1.AUTO_REUSE):
      d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
      Y_hat = tf.compat.v1.layers.dense(d_outputs, 1, activation=None) 
    return Y_hat

# ------------------------------------------------------------------------------------

  # Embedder & Recovery
  H = embedder(X, T)
  X_tilde = recovery(H, T)
    
  # Supervisor
  X_tilde_supervise = supervisor(X_tilde, T)
    
  # Synthetic data
  X_hat_unsupervised = recovery(Z, T)
  X_hat = supervisor(X_hat_unsupervised, T)

    
  # Discriminator
  Y_fake = discriminator(H, T)
  Y_real = discriminator(Z, T)     
    
  # Variables        
  e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
  s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]

# ------------------------------------------------------------------------------------

  # Discriminator loss
  D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
  D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)

  D_loss = D_loss_real + D_loss_fake
            
  # Autoencoder loss

  # 1. Reconstruction loss
  R_loss = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
  R_loss_joint = tf.compat.v1.losses.mean_squared_error(X, X_tilde) + tf.compat.v1.losses.mean_squared_error(X, X_tilde_supervise)

  # 2. Adversarial loss
  Ad_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    
  # 3. Supervised loss
  S_loss = tf.compat.v1.losses.mean_squared_error(X_tilde[:,1:,:], X_tilde_supervise[:,:-1,:]) + tf.compat.v1.losses.mean_squared_error(X_tilde[:,2:,:], X_tilde_supervise[:,:-2,:])
    
  # 4. Two Momments
  std_loss = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(Z,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(H,[0])[1] + 1e-6)))
  mean_loss = tf.reduce_mean(tf.abs((tf.nn.moments(Z,[0])[0]) - (tf.nn.moments(H,[0])[0])))

  Distribution_loss = std_loss + mean_loss
    
  # 5. Summation
  AE_loss =  R_loss_joint +  Ad_loss + Distribution_loss + S_loss
              
# ------------------------------------------------------------------------------------

  # optimizer
  AE_R_solver = tf.compat.v1.train.AdamOptimizer().minimize(R_loss, var_list = e_vars + r_vars)
  AE_solver = tf.compat.v1.train.AdamOptimizer().minimize(AE_loss, var_list = e_vars + r_vars + s_vars)
  D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
  S_solver = tf.compat.v1.train.AdamOptimizer().minimize(S_loss, var_list = r_vars + s_vars)   
        
  ## AVATAR training   
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
    
  # 1. Embedding network training
  print('Start Embedding Network Training')
    
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
    # Train embedder        
    _, step_e_loss = sess.run([AE_R_solver, R_loss], feed_dict={X: X_mb, T: T_mb})        
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) ) 
      
  print('Finish Embedding Network Training')
    
  # 2. Training only with supervised loss
  print('Start Training with Supervised Loss Only')
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)    
    # Random vector generation   
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    # Train generator       
    _, step_g_loss_s = sess.run([S_solver, S_loss], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})       
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )
      
  print('Finish Training with Supervised Loss Only')
    
  # 3. Joint Training
  print('Start Joint Training')
  
  for itt in range(iterations):
    # AAE training (twice more than discriminator training)
    for kk in range(2):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)               
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Train generator
      _, reconstruction, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([AE_solver, R_loss_joint, Ad_loss, S_loss, Distribution_loss], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
           
    # Discriminator training        
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
    # Random vector generation
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    # Check discriminator loss before updating
    check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    # Train discriminator (only when the discriminator does not work well)
    if (check_d_loss > 0.15):        
      _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
      
    # Print multiple checkpoints
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + 
            ', D_loss: ' + str(np.round(step_d_loss,4)) + 
            ', R_loss_: ' + str(np.round(reconstruction,4)) + 
            ', Ad_loss_: ' + str(np.round(step_g_loss_u,4)) + 
            ', S_loss_: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
            ', Distribution_loss: ' + str(np.round(step_g_loss_v,4)) )
  print('Finish Joint Training')

# ------------------------------------------------------------------------------------

  ## Synthetic data generation


  if num_samples == "same":
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)

  else:
    Z_mb = random_generator(num_samples, z_dim, ori_time, max_seq_len)
  
  generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})    
    
  generated_data = list()

  for i in range(no):
    temp = generated_data_curr[i,:ori_time[i],:]
    generated_data.append(temp)

  # Renormalization
  generated_data = generated_data * max_val
  generated_data = generated_data + min_val
  return generated_data

  

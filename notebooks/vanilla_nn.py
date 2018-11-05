#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('load_ext', 'autotime')
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
import matplotlib
import matplotlib.pyplot as plt


# In[6]:


import sys
sys.path.extend(["../"])


# In[7]:


from sklearn.model_selection import train_test_split
from core.tools.data_import import *
from core.tools.time_series import *
from constants import *


# In[9]:


df = load_dataset(UNRATE_DIR["MAC"])
df_d1 = differencing(df, periods=1, order=1)

lags = list(range(1, 25))
X_FEATURES = len(lags)
X, y = gen_supervised(df_d1, predictors=lags)
X, y = clean_nan(X, y)


# In[10]:


X.head()


# In[11]:


y.head()


# In[12]:


(X_train, X_test,
 y_train, y_test) = train_test_split(
    X, y,
    test_size=0.1,
    shuffle=False
)

(X_train, X_val,
 y_train, y_val) = train_test_split(
    X_train, y_train,
    test_size=0.1,
    shuffle=False
)


# In[13]:


print(f"Training and testing set generated,\nX_train shape: {X_train.shape}\ny_train shape: {y_train.shape}\nX_test shape: {X_test.shape}\ny_test shape: {y_test.shape}\nX_validation shape: {X_val.shape}\ny_validation shape: {y_val.shape}")


# In[14]:


plt.figure(figsize=(15,4))
plt.plot(y_train, linewidth=0.8, alpha=0.7)
plt.plot(y_test, linewidth=0.8, alpha=0.7)
plt.plot(y_val, linewidth=0.8, alpha=0.7)
plt.grid(True)
plt.legend(["Training", "Testing", "Validation"], loc="best")
plt.title("Training, Validation and Testing Target")
plt.show()


# In[15]:


INPUT_LAYER_SIZE = X_FEATURES
HIDDEN_LAYER_SIZE_1 = 512
HIDDEN_LAYER_SIZE_2 = 256
OUTPUT_LAYER_SIZE = 1
LEARNING_RATE = 0.003

EPOCHS = 1000
LOG_SEP = 10  # Log epoch seperation
REPORT_SEP = 100  # Report epoch seperation


# In[16]:


with tf.variable_scope("inputs"):
    X = tf.placeholder(
        dtype=tf.float32,
        shape=[None, X_FEATURES],
        name="predictors")
    y_true = tf.placeholder(
        dtype=tf.float32,
        shape=[None, 1],
        name="Response"
    )


# In[17]:


with tf.variable_scope("layers"):
    W1 = tf.Variable(tf.random_uniform(
        [INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE_1], -1, 1))
    b1 = tf.Variable(tf.random_uniform(
        [HIDDEN_LAYER_SIZE_1], -1, 1))
    
    W2 = tf.Variable(tf.random_uniform(
        [HIDDEN_LAYER_SIZE_1, HIDDEN_LAYER_SIZE_2], -1 ,1))
    b2 = tf.Variable(tf.random_uniform(
        [HIDDEN_LAYER_SIZE_2], -1, 1))
    
    W3 = tf.Variable(tf.random_uniform(
        [HIDDEN_LAYER_SIZE_2, OUTPUT_LAYER_SIZE], -1, 1))
    b3 = tf.Variable(tf.random_uniform(
        [OUTPUT_LAYER_SIZE], -1, 1))

    # HL: Hidden Layer
    # AHL: Activated Hidden Layer
    HL_1 = tf.add(tf.matmul(X, W1), b1)
    AHL_1 = tf.sigmoid(HL_1)
    
    HL_2 = tf.add(
        tf.matmul(AHL_1, W2), b2)
    
    AHL_2 = tf.sigmoid(HL_2)
    
    output_layer = tf.add(tf.matmul(AHL_2, W3), b3)

    loss = tf.reduce_mean(tf.square(y_true - output_layer))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)


# In[18]:


loss_history = {
    "epoch": list(),
    "train": list(),
    "val": list()
}

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(EPOCHS):
        sess.run(optimizer, feed_dict={
                    X: X_train.values,
                    y_true: y_train.values})
        
        if e % LOG_SEP == 0:
            train_loss = sess.run(loss, feed_dict={
                X: X_train.values,
                y_true: y_train.values
            })
            val_loss = sess.run(loss, feed_dict={
                X: X_val.values,
                y_true: y_val.values
            })
            loss_history["epoch"].append(e)
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            
        if e % REPORT_SEP == 0:
            train_loss = sess.run(loss, feed_dict={
                X: X_train.values,
                y_true: y_train.values
            })
            val_loss = sess.run(loss, feed_dict={
                X: X_val.values,
                y_true: y_val.values
            })
            print(f"Epoch {e}: Training Loss :{train_loss:0.7f}, Validation Loss :{val_loss:0.7f}")
    train_loss = sess.run(loss, feed_dict={X: X_train.values, y_true: y_train.values})
    val_loss = sess.run(loss, feed_dict={X: X_val.values, y_true: y_val.values})
    print(f"Final: Training Loss :{train_loss:0.7f}, Validation Loss :{val_loss:0.7f}")
    
    pred_train = sess.run(output_layer, feed_dict={
        X: X_train.values,
        y_true: y_train.values
    })
    
    pred_val = sess.run(output_layer, feed_dict={
        X: X_val.values,
        y_true: y_val.values
    })

    pred_test = sess.run(output_layer, feed_dict={
        X: X_test.values,
        y_true: y_test.values
    })


# In[19]:


pred_train = pd.DataFrame(pred_train)
pred_train.set_index(y_train.index, inplace=True)

pred_val = pd.DataFrame(pred_val)
pred_val.set_index(y_val.index, inplace=True)

pred_test = pd.DataFrame(pred_test)
pred_test.set_index(y_test.index, inplace=True)


# In[20]:


plt.close()
plt.figure(figsize=(45, 10))
for s in [y_train, pred_train,
          y_val, pred_val,
          y_test, pred_test]:
    plt.plot(s, linewidth=0.8, alpha=0.7)
plt.legend([
    "Training Set: Actual",
    "Training Set: Predicted",
    "Validation Set: Actual",
    "Validation Set: Predicted",
    "Testing Set: Actual",
    "Testing Set: Predicted"
])
plt.title("Training result")
plt.grid(True)
plt.show()


# In[21]:


plt.close()
plt.figure(figsize=(15, 5))
plt.plot(loss_history["epoch"], np.log(loss_history["train"]), linewidth=0.8)
plt.plot(loss_history["epoch"], np.log(loss_history["val"]), linewidth=0.8)
plt.legend(["Training Loss", "Validation Loss"])
plt.grid(True)
plt.show()


# In[ ]:





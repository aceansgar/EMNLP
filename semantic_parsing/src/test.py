from IPython import embed
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np
import tensorflow as tf
from sklearn.utils.fixes import logsumexp
import numpy as np

mean=tf.constant([[0.0,0.1,0.2],[0.3,0.4,0.5]])
a=tf.constant([[1.0,1.0,1.0],[1.0,1.0,1.0]])
variance=tf.Variable(a+1)
distribution=tf.distributions.Normal(loc=mean,scale=tf.sqrt(variance))
vec=[1,2,3]
cond_log_p=distribution.log_prob(vec)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(cond_log_p))





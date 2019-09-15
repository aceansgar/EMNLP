import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

const1=tf.constant(1)
var1=tf.Variable(initial_value=-1,name="var1_another")
var2=tf.Variable(initial_value=-1,name="var1")


a=tf.constant(100)
b=tf.constant(200)
var1=tf.Variable(a+b)


sess0=tf.Session()
sess0.run(tf.global_variables_initializer())
print(sess0.run(var1))
print(sess0.run(var2))


# print(sess0.run(const1))
# print(sess0.run(var1))
save_path="saves/var1.ckpt"
saver=tf.train.Saver({"var1":var1})
saver.save(sess0,save_path)



# Add ops to save and restore all the variables.


# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "/tmp/model.ckpt")
#   print("Model restored.")
#   # Check the values of the variables
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())
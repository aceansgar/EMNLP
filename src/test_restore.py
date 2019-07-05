import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

const1=tf.constant(1)
var3=tf.Variable(initial_value=-100,name="var1_another")



sess0=tf.Session()
sess0.run(tf.global_variables_initializer())

print(sess0.run(var3))

# print(sess0.run(const1))
# print(sess0.run(var1))
save_path="saves/var1.ckpt"
saver=tf.train.Saver({"var1":var3})
saver.restore(sess0,save_path)
print(sess0.run(var3))



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
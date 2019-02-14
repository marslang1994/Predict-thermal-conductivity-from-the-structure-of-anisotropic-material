import numpy as np
import pandas as pd
import tensorflow as tf
from ConvNetFunction import *
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import glob
import re

plt.switch_backend('agg') 

session = tf.Session()
batch_size = 100

image_size = 100
d = {}

image_path = "C:/Users/marsl/Desktop/ThermalConductiviy/ThermalConductiviy/Anisotropic/anisotropic_porous_media/image/2D_*.jpg"
addrs = glob.glob(image_path)
#######
label_index = []
for fname in addrs:
	res = re.findall("2D_([0-9]+).jpg", fname)
	if not res:
		print("This example does not have an image number: {}".format(fname))
	label_index.append(res)

# already sure that the order of images is the same in addrs and label_index

label_index = [i for j in label_index for i in j]
label_index = map(lambda x: int(x), label_index)
label_index = list(label_index)
label_file = pd.read_excel("C:/Users/marsl/Desktop/ThermalConductiviy/ThermalConductiviy/Anisotropic/anisotropic_porous_media/data.xlsx",
							sheet_name = "total")

series = []
for i in label_index:
	df = label_file.loc[label_file["image_num"] == i]
	series.append(list(df["Thermal conductivity(W/mK)"]))
series = [j for i in series for j in i]                            
series = np.asarray(series)

#with open('image-labels.txt', 'r') as f:
#	for line in f:
#		(key, val) = line.split()
#		d[str(key)] = float(val)
#d.pop('01840.jpg') # those files do not exit in image directory
#d.pop('02604.jpg')
#filenames = sorted(d.keys())

#labels = []
#for i in filenames:
#	labels.append(d[i])
#imageDir = './image'

#labels = np.asarray(labels)
imageDir = 'C:/Users/marsl/Desktop/ThermalConductiviy/ThermalConductiviy/Anisotropic/anisotropic_porous_media/image'
def load_images(path):
  #path = os.path.join(imageDir, filename)
  image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)# load image data in greyscale.
  image = cv2.resize(image, (100, 100),0,0,cv2.INTER_LINEAR)
  image = np.multiply(image, 1.0/255.0)#normalization
  image = image.reshape(image_size, image_size, 1)
  return image

img = []
for i in range(len(series)):
  img.append(load_images(addrs[i]))
 

images = np.array(img)
series = series.reshape(-1,1)
train_X, test_X, train_y, test_y = train_test_split(images, series, test_size=0.2, random_state = 42)



class DataSet(object):

  def __init__(self, images, labels):
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end] 


def read_train_sets(validation_size):
  class DataSets(object): # what does this do?????
    pass
  data_sets = DataSets()

  images, labels = train_X, train_y 

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
 
  train_images = images[validation_size:]
  train_labels = labels[validation_size:]

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.valid = DataSet(validation_images, validation_labels)

  return data_sets


data = read_train_sets(0.2)



# create placeholders for input data (both images and their labels).
X = tf.placeholder(tf.float32, shape = [None, image_size, image_size, 1], name = 'X')
y_true = tf.placeholder(tf.float32, shape = [None, 1], name = 'y')

#build the convnet
layer1 = conv_layer(input = X,
            channels = 1,
            kernel_size = 3,
            num_filters = 96)
layer2 = conv_layer(input = layer1,
          channels = 96,
          kernel_size = 3,
          num_filters = 96)
layer3 = conv_layer(input = layer2,
          channels = 96,
          kernel_size = 3,
          num_filters = 128)
layer4 = conv_layer(input = layer3,
          channels = 128,
          kernel_size = 3,
          num_filters = 96)
# flatten the data
flatten_op, nodes = flatten(layer4)
layer5 = fc_layer(input = flatten_op,
          num_inputs = nodes,
          num_outputs = 1024)
drop_out1 = tf.nn.dropout(x = layer5, keep_prob = 0.5)
layer6 = fc_layer(input = drop_out1,
          num_inputs = 1024,
          num_outputs = 512)
drop_out2 = tf.nn.dropout(x = layer6, keep_prob = 0.5)
output = fc_layer(input = layer6,
          num_inputs = 512,
          num_outputs = 1,
          use_activiation = False)
output = tf.identity(output, name = 'output')


# define the loss function
loss = tf.losses.mean_squared_error(labels = y_true, predictions = output)
# define the optimizer

optimizer = tf.train.AdamOptimizer().minimize(loss)
#optimizer = tf.keras.optimizers.SGD(lr=0.1).get_updates(loss)

session.run(tf.global_variables_initializer())

# define the process visualization
def show_process(epoch, feed_dict_train,  val_loss, train_error):
    train_loss = session.run(loss, feed_dict = feed_dict_train)
    train_error.append(train_loss)
    msg = 'Train epoch {0} --- Training Loss: {1: .3f}, Validaiton Loss: {2: .3f}' 
    print (msg.format(epoch + 1, train_loss, val_loss))


saver = tf.train.Saver()
train_error = []
validation_error = []
total_iterations = 0



def train(steps):
    global total_iterations

    for i in range (total_iterations, total_iterations + steps):
        X_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {X: X_batch, y_true: y_true_batch}

        val_x_batch, val_y_batch = data.valid.next_batch(batch_size)
        feed_dict_validation = {X: val_x_batch, y_true: val_y_batch}

        session.run(optimizer, feed_dict = feed_dict_train)
        if i % int(data.train.num_examples/batch_size) == 0:
          val_loss = session.run(loss, feed_dict = feed_dict_validation)
          validation_error.append(val_loss)
          epoch = int(i/int(data.train.num_examples/batch_size))
          show_process(epoch, feed_dict_train, val_loss, train_error)
          #saver.save(session, './trained_model')
        # if epoch == 199:
        #   break
train(1000)
# save train and validation error to files.
train_er = open('training_error5.txt', 'w')
val_er = open('val_er5.txt','w')

for i in train_error:
   train_er.write('%f\n' %i)
for j in validation_error:
   val_er.write('%f\n' %j) 

test_dict = {X: test_X}

out = session.run(output, feed_dict = test_dict)

mae = mean_absolute_error(test_y, out)
r2 = r2_score(test_y, out)
print ('Mean Abosolute Error is: %.3f' %mae,
        'R2 score is: %.3f' %r2)

weights =  session.run(tf.trainable_variables())[0]
np.save('weights.npy', weights)

'''
scatter plot
'''
plt.scatter(out, test_y)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.savefig('scatt_0.jpg', format = 'jpg', dpi = 300)














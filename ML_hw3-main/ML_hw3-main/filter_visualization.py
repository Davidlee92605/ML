import os
import numpy as np
import cv2
import keras
from keras import activations
from vis.utils import utils
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from vis.visualization import get_num_filters
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def feature_visualization(model):
	index = utils.find_layer_idx(model , 'dense_5')
	model.layers[index].activation = activations.linear
	model = utils.apply_modifications(model)

	for (i , layer) in enumerate(['conv2d_11']):
		fig = plt.figure(figsize = (18 , 2))

		index = utils.find_layer_idx(model , layer)
		number_of_filter = get_num_filters(model.layers[index])
		for j in range(20):
			image = visualize_activation(model , index , filter_indices = j , max_iter = 2000 , input_modifiers = [Jitter(15)])
			image = image[ : , : , 0]

			ax = fig.add_subplot(1 , 20 , j + 1)
			ax.set_axis_off()
			ax.imshow(image , cmap = 'gray')

		fig.suptitle('convolution layer {}'.format(i + 1) , fontsize = 24)
		plt.show()

	return

def main():
	model = keras.models.load_model('keras_model.h5')
	model = Model(inputs=model.inputs , outputs=model.layers[8].output)
	model.summary()
	img = cv2.imread('19637.jpg', cv2.IMREAD_GRAYSCALE)
	img = np.array(img)
	img = img.reshape(-1 , 48 , 48 , 1).astype('float')/255
	print(img.shape)
	feature_maps = model.predict(img)
	edg_1 = 8
	edg_2 = 8
	ix = 1
	for _ in range(8):
		for _ in range(8):
			# specify subplot and turn of axis
			ax = plt.subplot(edg_1, edg_2, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
			ix += 1
	plt.show()
	return

if (__name__ == '__main__'):
	main()
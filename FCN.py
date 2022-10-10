import os
import zipfile
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, BatchNormalization, ReLU, Concatenate, Softmax
# import BatchNormalization and ReLU
from tensorflow.keras.layers import BatchNormalization, ReLU
 
import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import load_image
import glob
import cv2
# cv2 pip install opencv-python
import random
from keras.preprocessing.image import ImageDataGenerator

# download the dataset (fcnn-dataset.zip)

# pixel labels in the video frames
class_names = ['sky', 'building','column/pole', 'road', 
               'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'bicyclist', 'void']


train_image_path = 'D:\study_data\_data\dataset1/images_prepped_train/'
train_label_path = 'D:\study_data\_data\dataset1/annotations_prepped_train/'
test_image_path = 'D:\study_data\_data\dataset1/images_prepped_test/'
test_label_path = 'D:\study_data\_data\dataset1/annotations_prepped_test/'
 
BATCH_SIZE = 16



def map_filename_to_image_and_mask(t_filename, a_filename, height=224, width=224):
    '''
    Preprocesses the dataset by:
        * resizing the input image and label maps
        * normalizing the input image pixels
        * reshaping the label maps from (height, width, 1) to (height, width, 12)
    
    Args:
        t_filename(string) -- path to the raw input image
        a_filename(string) -- path to the raw annotation (label map) file
        height(int) -- height in pixels to resize to
        width(int) -- width in pixels to resize to
    
    Returns:
        image(tensor) -- preprossed image
        annotation(tensor) -- preprocessed annotation
    '''
 
    # Convert image and mask files to tensors
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)
 
    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1,))
    stack_list = []
 
    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:,:,0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))
    
    annotation = tf.stack(stack_list, axis=2)
 
    # Normalize pixels in the input image
    image = image / 127.5 # normalize to [-1, 1]
    image -= 1
 
    return image, annotation
 
def get_dataset_slice_paths(image_dir, label_map_dir):
    '''
    generates the lists of image and label map paths
  
    Args:
        image_dir (string) -- path to the input images directory
        label_map_dir (string) -- path to the label map directory
 
    Returns:
        image_paths (list of strings) -- paths to each image file
        label_map_paths (list of strings) -- paths to each label map
    '''
 
    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_list]
 
    return image_paths, label_map_paths
 
def get_training_dataset(image_paths, label_map_paths):
    '''
    Prepares shuffled batches of the training set.
  
    Args:
        image_dir (string) -- path to the input images directory
        label_map_dir (string) -- path to the label map directory
 
    Returns:
        tf Dataset containing the preprocessed train set
    '''
    training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    training_dataset = training_dataset.map(map_filename_to_image_and_mask)
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)
 
    return training_dataset
 
def get_validation_dataset(image_paths, label_map_paths):
    '''
    Prepares shuffled batches of the validation set.
  
    Args:
        image_dir (string) -- path to the input images directory
        label_map_dir (string) -- path to the label map directory
 
    Returns:
        tf Dataset containing the preprocessed train set
    '''
    validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths)) # create a dataset from the image and label map paths
    validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.repeat()
 
    return validation_dataset


# get the paths to the images
training_image_paths, training_label_map_paths = get_dataset_slice_paths(train_image_path, train_label_path)
validation_image_paths, validation_label_map_paths = get_dataset_slice_paths(test_image_path, test_label_path)
# generate the train and valid sets
training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

# generate a list that contains one color for each class
colors = sns.color_palette(None, len(class_names)) # None
 
# print class name - normalized RGB tuple pairs
# the tuple values will be multiplied by 255 in the helper functions later
# to convert to the (0,0,0) to (255,255,255) RGB values you might be familiar with
for class_name, color in zip(class_names, colors): # zip the class names and colors together
    print(f'{class_name} -- {color}')
    
    
# Visualization Utilities
 
def fuse_with_pil(images):
    '''
    Creates a blank image and pastes input images
 
    Args:
        images (list of numpy arrays) - numpy array representations of the images to paste
  
    Returns:
        PIL Image object containing the images
    '''
 
    widths = (image.shape[1] for image in images)
    heights = (image.shape[0] for image in images)
    total_width = sum(widths)
    max_height = max(heights)
 
    new_im = PIL.Image.new('RGB', (total_width, max_height))
 
    x_offset = 0
    for im in images:
        pil_image = PIL.Image.fromarray(np.uint8(im))
        new_im.paste(pil_image, (x_offset,0))
        x_offset += im.shape[1]
  
    return new_im
 
 
def give_color_to_annotation(annotation):
    '''
    Converts a 2-D annotation to a numpy array with shape (height, width, 3) where
    the third axis represents the color channel. The label values are multiplied by
    255 and placed in this axis to give color to the annotation
 
    Args:
        annotation (numpy array) - label map array
  
    Returns:
        the annotation array with an additional color channel/axis
    '''
    seg_img = np.zeros( (annotation.shape[0],annotation.shape[1], 3) ).astype('float')
  
    for c in range(12):
        segc = (annotation == c)
        seg_img[:,:,0] += segc*( colors[c][0] * 255.0)
        seg_img[:,:,1] += segc*( colors[c][1] * 255.0)
        seg_img[:,:,2] += segc*( colors[c][2] * 255.0)
  
    return seg_img
 
 
def show_predictions(image, labelmaps, titles, iou_list, dice_score_list):
    '''
    Displays the images with the ground truth and predicted label maps
 
    Args:
        image (numpy array) -- the input image
        labelmaps (list of arrays) -- contains the predicted and ground truth label maps
        titles (list of strings) -- display headings for the images to be displayed
        iou_list (list of floats) -- the IOU values for each class
        dice_score_list (list of floats) -- the Dice Score for each vlass
    '''
 
    true_img = give_color_to_annotation(labelmaps[1])
    pred_img = give_color_to_annotation(labelmaps[0])
 
    image = image + 1
    image = image * 127.5
    images = np.uint8([image, pred_img, true_img])
 
    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
  
    display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score) for idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list) 
 
    plt.figure(figsize=(15, 4))
 
    for idx, im in enumerate(images):
        plt.subplot(1, 3, idx+1)
        if idx == 1:
            plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(im)
 
 
def show_annotation_and_image(image, annotation):
    '''
    Displays the image and its annotation side by side
 
    Args:
        image (numpy array) -- the input image
        annotation (numpy array) -- the label map
    '''
    new_ann = np.argmax(annotation, axis=2)
    seg_img = give_color_to_annotation(new_ann)
  
    image = image + 1
    image = image * 127.5
    image = np.uint8(image)
    images = [image, seg_img]
  
    images = [image, seg_img]
    fused_img = fuse_with_pil(images)
    plt.imshow(fused_img)
 
 
def list_show_annotation(dataset):
    '''
    Displays images and its annotations side by side
 
    Args:
        dataset (tf Dataset) - batch of images and annotations
    '''
 
    ds = dataset.unbatch()
    ds = ds.shuffle(buffer_size=100)
 
    plt.figure(figsize=(25, 15))
    plt.title("Images And Annotations")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)
 
    # we set the number of image-annotation pairs to 9
    # feel free to make this a function parameter if you want
    for idx, (image, annotation) in enumerate(ds.take(9)):
        plt.subplot(3, 3, idx + 1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(image.numpy(), annotation.numpy())
        
# print(class_names) # result = 
# print(colors) #result = 


def conv_block(inputs, filters, kernel_size, strides, padding='same'): # padding='same' or 'valid'
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def deconv_block(inputs, filters, kernel_size, strides, padding='same'):
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

# Vgg16.trainable = False # freeze the Vgg16

# nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, dilations=None, name=None) kernel_size default = 3
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

def FCN_8s():
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    x = model.output
    x = conv_block(x, 4096, 7, 1)
    x = conv_block(x, 4096, 1, 1)
    x = conv_block(x, 12, 1, 1)
    block5_conv1 = model.get_layer('block4_pool').output
    block5_conv1 = conv_block(block5_conv1, 12, 1, 1)
    block4_conv1 = model.get_layer('block3_pool').output
    block4_conv1 = conv_block(block4_conv1, 12, 1, 1)
    x = deconv_block(x, 12, 4, 2)
    concat1 = tf.keras.layers.Concatenate()([x, block5_conv1])
    x = deconv_block(concat1, 12, 4, 2)
    concat2 = tf.keras.layers.Concatenate()([x, block4_conv1])
    # 8x upsampling
    x = deconv_block(concat2, 12, 16, 8)
    x = tf.keras.layers.Softmax()(x)
    # x = deconv_block(concat2, 128, 16, 8)
    # x = tf.keras.layers.Conv2D(1, 1, 1, padding='same', activation='softmax')(x)
    model = tf.keras.Model(inputs=model.input, outputs=x)
    return model


train_count = len(training_image_paths)
validation_count = len(validation_image_paths)

steps_per_epoch = train_count // BATCH_SIZE
validation_steps = validation_count // BATCH_SIZE


fcnn = FCN_8s()
fcnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fcnn.fit(training_dataset, epochs=1, validation_data=validation_dataset, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

# # save the model
# fcnn.save('FCN_8s.h5')

# load the model
# fcnn = tf.keras.models.load_model('FCN_8s.h5')
fcnn.summary()

# predict the test image
pred = fcnn.predict(validation_dataset, steps=validation_steps, verbose=1)

my_image_path1 = 'D:/636E5906-FD2F-4C2E-BC7D-A257CA5BF180.png'
my_image_path2 = 'D:/D1AC37EF-52D2-4260-880C-A87B70BB49D1.png'

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0 # normalization
    image = np.expand_dims(image, axis=0)
    pred = fcnn.predict(image)
    pred = np.argmax(pred, axis=3)
    pred = np.squeeze(pred, axis=0)
    return pred

# iou
def iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union

# extract first image from validation dataset
image, annotation = next(iter(validation_dataset)) # (1, 224, 224, 3), (1, 224, 224, 1), next : get next element
image = image.numpy()
annotation = annotation.numpy()

score = iou(annotation[0], pred[0]) # 0.0
print(annotation[0].shape) # (224, 224, 1)
print(pred[0].shape) # (224, 224, 12)
exit()
print('iou score = ', score)

pred = np.argmax(pred, axis=3) # (batch_size, 224, 224)


# pred1 = predict_image(my_image_path1)
# pred2 = predict_image(my_image_path2)

def plot_image(image_path, pred):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(pred)
    plt.show()
    
# plot_image(my_image_path1, pred1)
# plot_image(my_image_path2, pred2)


# show 3 random images from the test set
for i in range(3):
    # get a random image from the test set
    image_index = np.random.randint(0, len(validation_image_paths)) # image.shape = (224, 224, 3)
    image = cv2.imread(validation_image_paths[image_index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
    # get the label map for the image
    label_map = cv2.imread(validation_label_map_paths[image_index], 0) # label_map.shape = (224, 224)
    # get the predicted label map for the image
    pred_label_map = pred[image_index]
    # plot the image, label map, and predicted label map
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].imshow(image)
    ax[1].imshow(label_map)
    ax[2].imshow(pred_label_map)
    ax[0].set_title('Image')
    ax[1].set_title('Label Map')
    ax[2].set_title('Predicted Label Map')
    plt.show()



# Model: "vgg16"
# _________________________________________________________________ 
# Layer (type)                Output Shape              Param #       
# ================================================================= 
# input_1 (InputLayer)        [(None, 256, 256, 3)]     0

#  block1_conv1 (Conv2D)       (None, 256, 256, 64)      1792        

#  block1_conv2 (Conv2D)       (None, 256, 256, 64)      36928       

#  block1_pool (MaxPooling2D)  (None, 128, 128, 64)      0

#  block2_conv1 (Conv2D)       (None, 128, 128, 128)     73856       

#  block2_conv2 (Conv2D)       (None, 128, 128, 128)     147584      

#  block2_pool (MaxPooling2D)  (None, 64, 64, 128)       0

#  block3_conv1 (Conv2D)       (None, 64, 64, 256)       295168      

#  block3_conv2 (Conv2D)       (None, 64, 64, 256)       590080      

#  block3_conv3 (Conv2D)       (None, 64, 64, 256)       590080      

#  block3_pool (MaxPooling2D)  (None, 32, 32, 256)       0

#  block4_conv1 (Conv2D)       (None, 32, 32, 512)       1180160     

#  block4_conv2 (Conv2D)       (None, 32, 32, 512)       2359808     

#  block4_conv3 (Conv2D)       (None, 32, 32, 512)       2359808     

#  block4_pool (MaxPooling2D)  (None, 16, 16, 512)       0

#  block5_conv1 (Conv2D)       (None, 16, 16, 512)       2359808     

#  block5_conv2 (Conv2D)       (None, 16, 16, 512)       2359808     

#  block5_conv3 (Conv2D)       (None, 16, 16, 512)       2359808     

#  block5_pool (MaxPooling2D)  (None, 8, 8, 512)         0

# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
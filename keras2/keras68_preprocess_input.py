from keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np 


model = ResNet50(weights='imagenet')

# img_path = "D:\study_data\_data\dog/cuteDog.jpg"
img_path = "D:\study_data\_data\dog/maker.jpg"
img = image.load_img(img_path, target_size=(224, 224))
print(img) #<PIL.Image.Image image mode=RGB size=224x224 at 0x2362E6C9280>

x = image.img_to_array(img)
print("================================= image.img_to_array(img) ==========================================")
print(x, '\n', x.shape) # (224, 224, 3)

x = np.expand_dims(x, axis=0) # expand_dims : 차원을 늘린다. # 0(1, 224, 224, 3)# 1 (224, 1, 224, 3) # (224, 224, 1, 3)
print("================================= np.expand_dims(x, axis=0) ==========================================")
print(x, '\n', x.shape) #  (1, 224, 224, 3)


x = preprocess_input(x)
print("=================================  preprocess_input(x)==========================================")
print(x, '\n', x.shape) #  (1, 224, 224, 3)
print(np.min(x),np.max(x))#

print("================================= (np.min(x,np.max(x)))==========================================")
print(x, '\n', x.shape) #  (1, 224, 224, 3)
print(np.min(x),np.max(x))#


print("=================================  preds = model.predict(x)=========================================")
preds = model.predict(x)
print(preds, "\n", preds.shape) # 

print("결과는 : ", decode_predictions(preds, top=5)[0]) #top=5 : 5개까지만 보여준다.

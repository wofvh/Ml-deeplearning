from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

#1
# model.trainable = False
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17

# 2
# for layes in model.layers:
#     layes.trainable = False
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17 
   

# model.layers[0].trainable = False # Dense 
# model.layers[1].trainable = False # Dense_1   
model.layers[2].trainable = False # Dense_2   

model.summary()

print(model.layers) # Dense 
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 11
# Non-trainable params: 6



# print(model.layers)# Dense_1  
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 9
# Non-trainable params: 8
 
 
# print(model.layers)# Dense_2  
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 14
# Non-trainable params: 3


#importing necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_folder = 'dataset/'

#calculates the average of the score for each images of the validaiton set
def compute_perf_metric(predictions, groundtruth):
    if predictions.shape == groundtruth.shape:
        return np.sum(predictions == groundtruth)/(predictions.shape[0]*predictions.shape[1])
    else:
        raise Exception('Error : the size of the arrays do not match. Cannot compute the performance metric')

#The vocublary set is the set of all the characters in the dataset
vocabulary = {'2','3','4','5','6','7','8','b','c','d','e','f','g','m','n','p','w','x','y'}

# The char_to_num dictionaries converts characters to integers and vice-versa
char_to_num = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'b':7,'c':8,'d':9,'e':10,'f':11,'g':12,'m':13,'n':14,'p':15,'w':16,'x':17,'y':18}

def encode_single_sample(img_path, label, crop):
    '''
    This function encodes a single sample. 
    Inputs :
    - img_path : the string representing the image path e.g. 'dataset/6n6gg.jpg'
    - label : the string representing the label e.g. '6n6gg'
    - crop : boolean, if True the image is cropped around the characters and resized to the original size.
    Outputs :
    - a multi-dimensional array reprensenting the image. Its shape is (50, 200, 1)
    - an array of integers representing the label after encoding the characters to integer. E.g [6,16,6,14,14] for '6n6gg' 
    '''
    
    img = tf.io.read_file(img_path) # Read image file and returns a tensor with dtype=string
    img = tf.io.decode_png(img, channels=1)     # This decode function returns a tensor with dtype=uint8
    img_ = tf.image.convert_image_dtype(img, tf.float32)  # Scales and returns a tensor with dtype=float32

    # Crop and resize to the original size 
    if(crop==True):
        img = tf.image.crop_to_bounding_box(img_, offset_height=0, offset_width=25, target_height=50, target_width=125)
        img = tf.image.resize(img,size=[50,200],method='bilinear', preserve_aspect_ratio=False,antialias=False, name=None)
   
    img = tf.transpose(img, perm=[1, 0, 2]) # Transpose the image because we want the time dimension to correspond to the width of the image.
    label = list(map(lambda x:char_to_num[x], label))  # Converts the string label into an array with 5 integers. E.g. '6n6gg' is converted into [6,16,6,14,14]
    return img.numpy(), label


def create_train_and_validation_datasets(crop=False):
    # Loop on all the files to create X whose shape is (1030, 50, 200, 1) and y whose shape is (1030, 5)
    X, y = [],[]
    for _, _, files in os.walk(img_folder):
        for f in files:
            # To start, let's ignore the jpg images
            label = f.split('.')[0]
            extension = f.split('.')[1]
            if extension=='png':
                img, label = encode_single_sample(img_folder+f, label,crop)
                X.append(img)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Split X, y to get X_train, y_train, X_val, y_val 
    X_train, X_val, y_train, y_val = train_test_split(X.reshape(1030, 10000), y, test_size=0.1, shuffle=True, random_state=42)
    X_train, X_val = X_train.reshape(927,200,50,1), X_val.reshape(103,200,50,1)
    return X_train, X_val, y_train, y_val

def build_model():
    
    # Inputs to the model
    input_img = layers.Input(shape=(200,50,1), name="image", dtype="float32") 

    # First conv block
    x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Reshape(target_shape=(5, 7680), name="reshape")(x)     # Reshape to "split" the volume in 5 time-steps

    # FC layers
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dense(64, activation="relu", name="dense2")(x)
   
    # Output layer
    output = layers.Dense(19, activation="softmax", name="dense3")(x) 
    
    # Define the model
    model = keras.models.Model(inputs=input_img, outputs=output, name="ocr_classifier_based_model")
    
    # Compile the model and return
    model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics="accuracy")
    return model


# Get the model
model = build_model()
model.summary()

#Train the model
X_train, X_val, y_train, y_val = create_train_and_validation_datasets(crop=True)
history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=30)


#model predictions
y_pred = model.predict(X_val) # y_pred shape = (104,50,19)
y_pred = np.argmax(y_pred, axis=2)
num_to_char = {'-1':'UKN','0':'2','1':'3','2':'4','3':'5','4':'6','5':'7','6':'8','7':'b','8':'c','9':'d','10':'e','11':'f','12':'g','13':'m','14':'n','15':'p','16':'w','17':'x','18':'y'}
nrow = 1
fig=plt.figure(figsize=(20, 5))
for i in range(0,10):
    if i>4: nrow = 2
    fig.add_subplot(nrow, 5, i+1)
    plt.imshow(X_val[i].transpose((1,0,2)),cmap='gray')
    plt.title('Prediction : ' + str(list(map(lambda x:num_to_char[str(x)], y_pred[i]))))
    plt.axis('off')
plt.show()

#model performance
print("The Model performance on validation data set is  : {}" .format(compute_perf_metric(y_pred, y_val)))

test_folder = 'test/'

def create_test_datasets(crop=False):
    # Loop on all the files to create X whose shape is (1040, 50, 200, 1) and y whose shape is (1040, 5)
    X, y = [],[]

    for _, _, files in os.walk(test_folder):
        for f in files:
            # To start, let's ignore the jpg images
            label = f.split('.')[0]
            extension = f.split('.')[1]
            if extension=='png':
                img, label = encode_single_sample(test_folder+f, label,crop)
                X.append(img)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y


X_test, y_test = create_test_datasets(crop=True)

y_predt = model.predict(X_test)
y_predt = np.argmax(y_predt, axis=2)
nrow = 1
fig=plt.figure(figsize=(20, 5))
for i in range(0,10):
    if i>4: nrow = 2
    fig.add_subplot(nrow, 5, i+1)
    plt.imshow(X_test[i].transpose((1,0,2)),cmap='gray')
    plt.title('Prediction : ' + str(list(map(lambda x:num_to_char[str(x)], y_predt[i]))))
    plt.axis('off')
plt.show() 

print("The Model performance on Test data set is  : {}" .format(compute_perf_metric(y_predt, y_test)))

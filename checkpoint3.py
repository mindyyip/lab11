# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import numpy

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


# New test images
images = ['pants.png', 'shirt.png', 'sweater.jpg']
edited_images = [ Image.open('images/' + i) for i in images]

# Scale and grey each image
for i in range(len(images)):
    new_image = ImageOps.grayscale(edited_images[i])
    new_image = new_image.resize((28, 28))
    new_image.save("edited_" + images[i])
    # new_image = ImageOps.invert(new_image)
    edited_images[i] = np.array(new_image)

edited_images = np.array(edited_images) / 255.0
labels = np.array([0]*len(images)) # random labels since we dont need them

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

# Evaluate custom images

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = model.predict(test_images)
predictions = model.predict(edited_images)

# Print predictions for each model
for i in range(len(images)):
    predicted_label = np.argmax(predictions[i])
    print(images[i])
    print(predictions[i], class_names[predicted_label])
# # TensorFlow and tf.keras
# import tensorflow as tf
#
# # Helper libraries
# from PIL import Image
# from PIL import ImageOps
# import numpy as np
# import matplotlib.pyplot as plt
#
# fashion_mnist = tf.keras.datasets.fashion_mnist
#
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
#
# original_images = ['pants.png', 'shirt.png', 'hat.png']
# edited_images = []
# for image in original_images:
#     new_images = Image.open('images/' + image)
#     new_image = ImageOps.grayscale(new_images)
#     new_image = new_image.resize((28,28))
#     new_image.save('edited_' + image)
#     # new_image = ImageOps.invert(new_image)
#     edited_images.append(np.array(new_image))
#
# # test_images = np.array(edited_images)
#
# train_images = train_images / 255.0
# test_images = test_images / 255.0
# edited_images = np.array(edited_images)/255.0
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.fit(train_images, train_labels, epochs=10)
#
# predictions = model.predict(test_images)
# predictions = model.predict(edited_images)
#
# for prediction in predictions:
#     print(prediction)
#     print(np.argmax(prediction[0]))
#     print(class_names[np.argmax(prediction[0])])

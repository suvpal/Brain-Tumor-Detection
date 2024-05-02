import os

base_dir = '/content/drive/MyDrive/ML datasets/brainTumorDataset'

print("Contents of base directory:")
print(os.listdir(base_dir))

print("\nContents of train directory:")
print(os.listdir(f'{base_dir}/Training'))

train_dir = os.path.join(base_dir, 'Training')

train_glioma_dir = os.path.join(train_dir, 'glioma')
train_meningioma_dir = os.path.join(train_dir, 'meningioma')
train_pituitary_dir = os.path.join(train_dir, 'pituitary')
train_notumor_dir = os.path.join(train_dir, 'notumor')

testing_dir = os.path.join(base_dir, 'Testing')

testing_glioma_dir = os.path.join(testing_dir, 'glioma')
testing_meningioma_dir = os.path.join(testing_dir, 'meningioma')
testing_pituitary_dir = os.path.join(testing_dir, 'pituitary')
testing_notumor_dir = os.path.join(testing_dir, 'notumor')

train_glioma_fnames = os.listdir(train_glioma_dir)
train_meningioma_fnames = os.listdir(train_meningioma_dir)
train_pituitary_fnames = os.listdir(train_pituitary_dir)
train_notumor_fnames = os.listdir(train_notumor_dir)

testing_glioma_fnames = os.listdir(testing_glioma_dir)
testing_meningioma_fnames = os.listdir(testing_meningioma_dir)
testing_pituitary_fnames = os.listdir(testing_pituitary_dir)
testing_notumor_fnames = os.listdir(testing_notumor_dir)

print('total training glioma images :', len(os.listdir(train_glioma_dir)))
print('total training meningioma images :', len(os.listdir(train_meningioma_dir)))
print('total training pituitary images :', len(os.listdir(train_pituitary_dir)))
print('total training notumor images :', len(os.listdir(train_notumor_dir)))
print('\n')
print('total training glioma images :', len(os.listdir(testing_glioma_dir)))
print('total training meningioma images :', len(os.listdir(testing_meningioma_dir)))
print('total training pituitary images :', len(os.listdir(testing_pituitary_dir)))
print('total training notumor images :', len(os.listdir(testing_notumor_dir)))

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size= 20,
                                                    shuffle = True,
                                                    class_mode ='categorical',
                                                    target_size=(300, 300))

validation_generator =  test_datagen.flow_from_directory(testing_dir,
                                                         batch_size= 20,
                                                         shuffle = True,
                                                         class_mode = 'categorical',
                                                         target_size = (300, 300))

history = model.fit(
            train_generator,
            epochs=4,
            validation_data=validation_generator,
            verbose=2)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
print(acc)
print()

print(val_acc)
epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

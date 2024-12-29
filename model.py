import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Verilerin konumları
text_images_folder = 'C:/Users/yusuf/PycharmProjects/pythonProject1/projeler/output/handwritten'
label_images_folder = 'C:/Users/yusuf/PycharmProjects/pythonProject1/projeler/output/labels'

# Görüntü boyutları
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Veri yükleme fonksiyonu
def load_images_from_folder(folder, label_value):
    images = []
    labels = []
    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)
        if filename.endswith('.png'):
            img = tf.keras.utils.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img)
            labels.append(label_value)
    return images, labels

# El yazısı görüntüler
text_images, text_labels = load_images_from_folder(text_images_folder, 0)

# Etiket görüntüler
label_images, label_labels = load_images_from_folder(label_images_folder, 1)

# Görüntüleri ve etiketleri birleştirme
images = np.array(text_images + label_images)  # NumPy dizisine dönüştür
labels = np.array(text_labels + label_labels)  # NumPy dizisine dönüştür

# Veriyi eğitim ve test setine bölme
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# CNN modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# Sonuçları değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")


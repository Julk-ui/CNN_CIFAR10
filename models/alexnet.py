from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

def AlexNet(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),

        # Primera capa convolucional
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),

        # Segunda capa convolucional
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),

        # Tercera capa convolucional (sin max pooling)
        Conv2D(256, (3, 3), activation='relu', padding='same'),

        # Flatten y capas densas
        Flatten(),
        Dense(512, activation='relu'),  # ⚡ Reducción de tamaño
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # ⚡ Eliminamos capas extras
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

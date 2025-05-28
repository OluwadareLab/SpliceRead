from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scripts.models.residual_block import ResidualBlock

def deep_cnn_classifier(sequence_length):
    model = Sequential()
    model.add(Conv1D(50, 9, strides=1, padding='same', input_shape=(sequence_length, 4), activation='relu'))
    for _ in range(3):
        model.add(ResidualBlock(50, 9))
        model.add(AveragePooling1D(pool_size=2, strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

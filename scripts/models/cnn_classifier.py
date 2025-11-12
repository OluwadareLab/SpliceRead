from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax
from tensorflow.keras.regularizers import l2
from scripts.models.residual_block import ResidualBlock

def deep_cnn_classifier(sequence_length, num_classes):
    model = Sequential()
    model.add(Conv1D(50, 9, strides=1, padding='same', input_shape=(sequence_length, 4), activation='relu'))
    for _ in range(3):
        model.add(ResidualBlock(50, 9))
        model.add(AveragePooling1D(pool_size=2, strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_with_hyperparameters(
    sequence_length, 
    num_classes,
    num_conv_layers=3,
    num_filters=50,
    kernel_size=9,
    dropout_rate=0.3,
    learning_rate=0.001,
    optimizer='adam',
    l2_reg=0.0,
    dense_units=100,
    use_batch_norm=False,
    **kwargs
):
    """
    Create a CNN model with configurable hyperparameters.
    
    Args:
        sequence_length: Input sequence length
        num_classes: Number of output classes
        num_conv_layers: Number of convolutional layers
        num_filters: Number of filters in conv layers
        kernel_size: Kernel size for conv layers
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop', 'adamax')
        l2_reg: L2 regularization strength
        dense_units: Number of units in dense layer
        use_batch_norm: Whether to use batch normalization
        **kwargs: Additional parameters (ignored)
    
    Returns:
        Compiled Keras model
    """
    
    model = Sequential()
    
    # Input layer
    model.add(Conv1D(
        num_filters, 
        kernel_size, 
        strides=1, 
        padding='same', 
        input_shape=(sequence_length, 4), 
        activation='relu',
        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
    ))
    
    if use_batch_norm:
        model.add(BatchNormalization())
    
    # Convolutional layers with residual blocks
    for i in range(num_conv_layers):
        model.add(ResidualBlock(num_filters, kernel_size))
        model.add(AveragePooling1D(pool_size=2, strides=1, padding='same'))
        
        if use_batch_norm:
            model.add(BatchNormalization())
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Choose optimizer
    if optimizer.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer.lower() == 'adamax':
        opt = Adamax(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)  # Default fallback
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load the IMDB dataset (10,000 most common words).
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding: Match review length (to 200 words).
max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test  = pad_sequences(X_test, maxlen=max_len)

# Define the model.
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    GlobalAveragePooling1D(),       # Reduces to a fixed vector.
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output.
])

# Compile the model.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training.
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# Test evaluation.
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy en test: {accuracy:.4f}")

# Graph loss curve.
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

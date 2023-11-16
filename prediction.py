from preprocess import preproc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


df = preproc('train.csv')
df.to_csv('train_preprocessed.csv', index=False)

X = df.drop(columns=['Transported'])  # Features (all columns except 'Transported')
y = df['Transported']  # Target variable

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_normalized = scaler.fit_transform(X_train)

# Use the same scaler to transform the validation data
X_val_normalized = scaler.transform(X_val)

model = Sequential()  # Create a Sequential model

# Adding layers to the model
# Using the ReLU activation function for the hidden layers
# Using the Sigmoid activation function for the output layer
# Also using Dropout layers for regularization
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Optional: Dropout layer for regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpointer=ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)

# nadam optomizer
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

history = model.fit(X_train_normalized, y_train, validation_data=(X_val_normalized, y_val), epochs=100, callbacks=[checkpointer, early_stopping])

loss, accuracy = model.evaluate(X_val, y_val)

print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


test_data = pd.read_csv('test.csv')

# used if the model is trained already
# model = load_model('weights.hdf5')


# preprocessing the test csv
pdd= preproc('test.csv')

pdd.to_csv('test_preprocessed.csv', index=False)

pdd_normalized = scaler.transform(pdd)

predictions = model.predict(pdd_normalized)
predictions=predictions.round().astype(int).astype(bool)

# Create a submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'].values.flatten(),  # Replace with the appropriate column name
    'Transported': predictions.flatten()  # Replace with the appropriate column name
})

# Save the submission to a CSV file
submission.to_csv('sample_submission.csv', index=False)

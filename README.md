# Doofstrom
import numpy as np
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH = 'G:\signbot\DATA\MP_DATA'
actions = np.array(open('G:\signbot\DATA\classes.txt', 'r').read().split('\n'))
number_of_sequences = 100
every_sequence_length = 30
sequences, labels = [], []

label_map = {label:num for num, label in enumerate(actions)}
label_map

for action in actions:
    for sequence in range(number_of_sequences):
        window = []
        for frame_num in range(every_sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            # print(res)
            except:
                pass
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# sequences[0]

X = np.array(sequences)

# X

# X.shape

y = to_categorical(labels).astype(int)

# y

# y.shape

X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X,y, train_size=0.9)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_and_val,y_train_and_val, test_size=0.2)

# X_train.shape

# X_valid.shape

# X_test.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

batch_size = 128    
epochs = 1000

# log_dir = 'Logs', datetime.datetime.now().strftime("%Y.%m.%d-%H.%M") + '--batch__' + str(batch_size) + '--epochs__' + str(epochs)
# tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_size, validation_data= (X_valid, y_valid))

model.summary()

print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=1)
print("test loss, test acc:", results)

print("Generate predictions for 30 samples")
predictions = model.predict(X_test)
print("predictions shape:", predictions.shape)

model.save('G:/signbot/MODEL/action1.h5')


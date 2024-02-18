from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import MeCab
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

mecab = MeCab.Tagger()

def filter_morphs(text):
    nodes = mecab.parseToNode(text)
    filtered_words = []
    while nodes:
        if nodes.feature.split(",")[0] in ["名詞", "動詞", "形容詞"]:
            filtered_words.append(nodes.surface)
        nodes = nodes.next
    return " ".join(filtered_words)

df = pd.read_csv('C:\\Users\\forev\\OneDrive\\바탕 화면\\asikan_240218.csv')
df['Index'] = df.index

titles = df['Title'].apply(filter_morphs).values
genres = df['Genre'].values
indices = df['Index'].values

# Analysis of Text Length
sequence_lengths = np.array([len(x) for x in titles])
maxlen = int(np.percentile(sequence_lengths, 95))

print(f"95% of the sequences have length <= {maxlen}. Setting maxlen={maxlen}.")

# Decide num_words by frequency of words.
word_freq = {}
for title in titles:
    for word in title.split():
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

num_words = len(word_freq)
print(f"Unique words in dataset: {num_words}")

# Text Tokenizer
tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(titles) # Create an index for words.
sequences = tokenizer.texts_to_sequences(titles) # Convert text data into sequences of integers.
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post') # Pad sequences to the same length.

# Label encoding and One-hot encoding
label_encoder = LabelEncoder()
genre_labels = label_encoder.fit_transform(genres) # Convert genre labels into integers.
genre_labels = to_categorical(genre_labels) # One-hot Encoding ([0. 0. 0. 0. 0. 0. 0. 1.])

X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(padded_sequences, genre_labels, indices, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    # input_dim: the number of unique words in the vocabulary.
    # output_dim: the dimension of the embedding vectors.
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=maxlen),
    tf.keras.layers.Conv1D(256, 5, activation='swish'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='swish'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.8),
    
    tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(genre_labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    epochs=1000,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping]
)

# 예측하기
predicted_genre_probabilities = model.predict(X_test)
predicted_genre_indices = np.argmax(predicted_genre_probabilities, axis=1)
predicted_genres = label_encoder.inverse_transform(predicted_genre_indices)

# 실제 장르 가져오기
true_genre_indices = np.argmax(Y_test, axis=1)
true_genres = label_encoder.inverse_transform(true_genre_indices)

# 검증 데이터셋의 제목 가져오기
validation_titles = df.loc[indices_test, 'Title']

# 출력
for i in range(10):
    print(f"Title: {validation_titles.iloc[i]}")
    print(f"Predicted Genre: {predicted_genres[i]}")
    print(f"Real Genre: {true_genres[i]}\n")
    
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save("asikan_classification_model")
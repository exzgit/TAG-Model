import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import os
import matplotlib.pyplot as plt

def scaled_dot_product_attention(queries, keys, values, mask=None):
    # Calculate the dot product
    product = tf.matmul(queries, keys, transpose_b=True)
    
    # Scale the dot product
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)
    
    # Apply mask if provided
    if mask is not None:
        scaled_product += (mask * -1e9)
    
    # Softmax calculation
    attention_weights = tf.nn.softmax(scaled_product, axis=-1)
    
    # Weighted sum
    output = tf.matmul(attention_weights, values)
    return output, attention_weights

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout = dropout
        
        # Splitting heads
        self.depth = self.head_size // self.num_heads
        self.wq = Dense(self.head_size)
        self.wk = Dense(self.head_size)
        self.wv = Dense(self.head_size)
        self.dense = Dense(self.head_size)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        q, k, v, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(q)[0]
        
        # Linear transformation
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        
        # Splitting heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # Concatenating heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.head_size))
        
        # Final linear transformation
        outputs = self.dense(concat_attention)
        return outputs, attention_weights

def positional_encoding(max_len, head_size):
    pos = tf.cast(tf.range(max_len)[:, tf.newaxis], dtype=tf.float32)
    i = tf.cast(tf.range(head_size)[tf.newaxis, :], dtype=tf.float32)
    angle_rads = pos / tf.pow(10000, (i-i%2)/head_size)
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    return tf.concat([sines, cosines], axis=-1)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, num_layers, dropout=0):
    inputs += positional_encoding(tf.shape(inputs)[1], head_size)
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder_layer(x, head_size, num_heads, ff_dim, dropout)
    return x

def transformer_encoder_layer(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head self-attention mechanism
    attention, _ = MultiHeadAttentionLayer(num_heads, head_size)(inputs={'query': inputs, 'key': inputs, 'value': inputs, 'mask': None})
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # Feed-forward neural network
    ffn = tf.keras.Sequential(
        [Dense(ff_dim, activation="relu"), Dense(inputs.shape[-1]),]
    )
    ffn_out = ffn(attention)
    ffn_out = Dropout(dropout)(ffn_out)
    return LayerNormalization(epsilon=1e-6)(attention + ffn_out)

def build_transformer_model(max_len, vocab_size, head_size, num_heads, ff_dim, num_layers, dropout=0.5):
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=head_size)(inputs)
    x = transformer_encoder(embedding_layer, head_size, num_heads, ff_dim, num_layers, dropout)
    x = Dense(vocab_size, activation="softmax")(x[:, -1, :])
    return Model(inputs=inputs, outputs=x)

dataset_path = "TAGModel/dataset/"
# Load dataset
conversations = []
# Iterate through files in the dataset directory
for file_name in os.listdir(dataset_path):
    # Check if the file is a text file
    if file_name.endswith('.txt'):
        file_path = os.path.join(dataset_path, file_name) # Construct the full path to the file
        # Open the file and append its content to the conversations list
        with open(file_path, 'r', encoding='utf-8') as file:
            conversations.append(file.read())

# Split the conversations into individual lines
conversations = [line for convo in conversations for line in convo.splitlines()]

# Tokenize the dataset
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False, filters='', lower=False, oov_token="[-START-] [-END-]")
tokenizer.fit_on_texts(conversations)
total_words = len(tokenizer.word_index) + 1

# Save the tokenizer for future use
tokenizer_path = "TAGModel/models/tokenizer.pkl"
with open(tokenizer_path, 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Generate input sequences for training
input_sequences = []
for line in conversations:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad input sequences for uniform length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Generate features and labels for training
X, y = input_sequences[:,:-1], input_sequences[:,-1]

# Define parameters
head_size = 512
num_heads = 8
ff_dim = 1024
num_layers = 12
dropout = 0.2

# Inisialisasi callback ModelCheckpoint
checkpoint_path = "TAGModel/checkpoint/"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

# Build and compile the transformer model
model = build_transformer_model(max_sequence_len - 1, total_words, head_size, num_heads, ff_dim, num_layers, dropout)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Train the model
history = model.fit(X, y, batch_size=64, epochs=50, validation_split=0.3, verbose=1, callbacks=[checkpoint_callback])

# Load the best weights
model.load_weights(checkpoint_path)

# Save the trained model
model.save("TAGModel/models/TAGModel")

# Predict function
def generate_text(model, tokenizer, initial_text, temperature=1.0, max_length=50):
    input_text = initial_text + " [-START-]"
    output_text = "[-START-] "
    for _ in range(max_length):
        tokenized_input = tokenizer.texts_to_sequences([input_text])[0]
        padded_input = pad_sequences([tokenized_input], maxlen=max_sequence_len-1, padding='pre')
        response = model.predict(padded_input)[0]

        # Adjust predictions with temperature scaling
        response = np.log(response) / temperature
        exp_response = np.exp(response)
        response = exp_response / np.sum(exp_response)

        # Sample the next token
        if output_text.startswith('[-START-]'):
            predicted_word_index = np.random.choice(len(response), p=response)
            predicted_word = tokenizer.index_word[predicted_word_index]
        
        if predicted_word == '[-END-]':
            break

        input_text += ' ' + predicted_word
        output_text += predicted_word + ' '      

    output_text = output_text.replace("[-START-]", "")



# Example usage
user_input = "Halo! Apa kabar?"
response = generate_text(user_input)
print("\033[92m" + "GENERATE >> " + "\033[0m" + " " + "\033[96m" + response + "\033[0m")


# Training visualization using scatter plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Training History')
plt.legend()
plt.savefig("TAGModel/image/1.jpeg")
plt.show()
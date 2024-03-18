import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import datetime

# Load tokenizer
with open("tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load model
model = load_model("TAGModel/models/TAGModel")

# Define max sequence length
max_sequence_len = model.input_shape[1] + 1


def generate_text(model, tokenizer, initial_text, temperature=1.0, max_length=50):
    input_text = initial_text + " [-START-]"
    output_text = "[-START-]"
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
            predicted_word = get_current_time()
            break

        input_text += ' ' + predicted_word
        output_text += predicted_word + ' '      

    output_text = output_text.replace("[-START-] ", "")


    return output_text

def get_current_time():
    current_time = datetime.datetime.now()
    return current_time

while True:
    prompt = input("\033[92m" + "PROMPT >> " + "\033[0m")
    response =  generate_text(model, tokenizer, prompt, temperature=0.7)
    print("\033[92m" + "GENERATE >> " + "\033[0m" + " " + "\033[96m" + response + "\033[0m")

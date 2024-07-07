import streamlit as st
import pandas as pd
import numpy as np
import time
from next_character_gen import decoder_transformer
import torch

# Set device for torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read and process text file for character encoding and decoding
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# Streamlit app
st.title('Shakespeare Text Generator')
st.write('This app generates text based on a given prompt using a decoder only transformer (similar to GPT mechanism).')

# Input prompt and number of tokens slider
input_txt = st.text_input('Enter a prompt from any Shakespeare text:')
st.write('⚠️ Note: Generation can take up to a minute.')
n_tokens = st.slider('Number of tokens to generate:', 1, 500, 200)

# Caching the model loading
@st.cache_resource  # Add the caching decorator
def load_model():
    model = decoder_transformer().to(device)
    model.load_state_dict(torch.load('trained_model_final.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model
model = load_model()

st.write("👇 Click the button below to generate text:")
# Trigger text generation with a button
def type_text(text):
    # Create an empty text element
    text_element = st.empty()
    s = ""
    for char in text:
        # Update the text element with the next character
        s += char
        text_element.write(s+'$ꕯ$')
        time.sleep(0.004)  # Adjust the sleep duration for the typing speed

    text_element.write(s)

if st.button('Generate Text'):
    if input_txt:
        st.subheader("Prompt")
        type_text(input_txt)
        context = torch.tensor(encode(input_txt), dtype=torch.long, device=device).unsqueeze(0)
        
        # Show loading spinner while generating text
        with st.spinner('Generating text...'):
            start_time = time.time()
            result = decode(model.generate(context, max_new_tokens=n_tokens)[0].tolist())
            end_time = time.time()
        
        # Display the generated result and the time taken
        st.subheader('Generated Text:')
        print(input_txt+result)
        type_text(input_txt+result)
        #st.write(result)
        st.write(f"Time taken: {end_time - start_time:.2f} seconds")
    else:
        st.warning('Please enter a prompt to generate text.')

# Add footer
st.markdown('---')
#st.markdown('Created with ❤️ using Streamlit and PyTorch')

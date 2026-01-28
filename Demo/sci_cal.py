# streamlit_scientific_calculator.py

import streamlit as st
import math

st.set_page_config(page_title="Scientific Calculator", layout="wide")

st.title("🔬 Scientific Calculator")

# Text input for expression
expr = st.text_input("Enter expression:", "")

# Buttons layout
buttons = [
    ['7', '8', '9', '/', 'C'],
    ['4', '5', '6', '*', '('],
    ['1', '2', '3', '-', ')'],
    ['0', '.', '^', '+', '='],
    ['sin', 'cos', 'tan', 'log', 'sqrt'],
    ['ln', 'pi', 'e', '', '']
]

# Store user input in session state
if 'expression' not in st.session_state:
    st.session_state.expression = ""

# Function to handle button click
def on_click(btn_text):
    if btn_text == "=":
        try:
            exp = st.session_state.expression
            # Replace function names with math equivalents
            exp = exp.replace('sin', 'math.sin')
            exp = exp.replace('cos', 'math.cos')
            exp = exp.replace('tan', 'math.tan')
            exp = exp.replace('log', 'math.log10')
            exp = exp.replace('ln', 'math.log')
            exp = exp.replace('sqrt', 'math.sqrt')
            exp = exp.replace('^', '**')
            result = eval(exp)
            st.session_state.expression = str(result)
        except Exception:
            st.session_state.expression = "Error"
    elif btn_text == "C":
        st.session_state.expression = ""
    elif btn_text == 'pi':
        st.session_state.expression += str(math.pi)
    elif btn_text == 'e':
        st.session_state.expression += str(math.e)
    else:
        st.session_state.expression += btn_text

# Display current expression
st.text_input("Expression:", value=st.session_state.expression, key="display", disabled=True)

# Create buttons
for row in buttons:
    cols = st.columns(len(row))
    for col, btn_text in zip(cols, row):
        if btn_text:
            col.button(btn_text, on_click=lambda b=btn_text: on_click(b))


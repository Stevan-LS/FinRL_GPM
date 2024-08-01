import streamlit as st
from solutions.GraphPortfolioManagement.test import test
from PIL import Image

def predict(initial_amount, num_weeks):
    #return f"Prediction for {num_weeks} week(s)"
    test(initial_amount)

def main():
    st.title("Financial copilot")
    
    num_weeks = st.slider("Select number of testing weeks", min_value=2, max_value=8, value=2)
    
    if st.button("Test"):
        predict(num_weeks, 10000)
        image = Image.open('plots.result_GPM.jpg')
        st.image(image, caption='Portfolio value evolution')

if __name__ == "__main__":
    main()
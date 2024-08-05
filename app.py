import streamlit as st
from test import test

def predict(initial_amount, num_weeks):
    #return f"Prediction for {num_weeks} week(s)"
    return test(initial_amount)

def main():
    st.title("Financial copilot")
    
    initial_amount = st.number_input("Enter an integer:", min_value=1000, max_value=1000000, step=1000, value=10000)
    num_weeks = st.slider("Select number of testing weeks", min_value=2, max_value=8, value=2)
    
    if st.button("Test"):
        graph = predict(initial_amount, num_weeks)
        #plotly_fig = tls.mpl_to_plotly(graph)
        st.pyplot(graph)

if __name__ == "__main__":
    main()
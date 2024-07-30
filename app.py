import streamlit as st

def predict(num_weeks):
    return f"Prediction for {num_weeks} week(s)"

def main():
    st.title("Predictor Interface")
    
    num_weeks = st.slider("Select number of testing weeks", min_value=2, max_value=8, value=2)
    
    if st.button("Test"):
        result = predict(num_weeks)
        st.success(result)

if __name__ == "__main__":
    main()
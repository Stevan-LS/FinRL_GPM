import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np

st.title("Test app")
# Sample data
days = np.linspace(0, 100, 11).astype(int)
values = np.sin(days)
# Create a Plotly figure
fig = go.Figure()

fig.add_trace(go.Scatter(x=days, y=values, mode='lines+markers', name='Values'))

# Show the plot in Streamlit
st.plotly_chart(fig)

# Displaying clicked point's value
selected_point = st.selectbox('Select a day:', days)

if selected_point:
    index = (days == selected_point).argmax()
    value = values[index]
    st.write(f'Value on {selected_point}: {value}')

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


timeframe = st.selectbox("Select Timeframe:", ["Daily", "Weekly", "Monthly", "Overall"])

daily_return = 0.000032
weekly_return= 0.000032
monthly_return= 0.000032
st.metric(label="Daily Return", value=f"{daily_return*100}%")
st.metric(label="Weekly Return", value=f"{weekly_return*100:.5f}%")
st.metric(label="Monthly Return", value=f"{monthly_return*100:.2f}%")
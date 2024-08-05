import streamlit as st
from test import test
import plotly.graph_objs as go
import numpy as np
import pandas as pd

def process(initial_amount):
    GPM_test, UBAH_test, portfolio_weights, asset_memory = test(initial_amount)

    # Build figure of portfolio value evolution
    fig = go.Figure()
    days = np.arange(1, len(GPM_test['values'])+1)
    fig.add_trace(go.Scatter(
        x=days, 
        y=GPM_test['values'], 
        mode='lines+markers', 
        name='GPM',
        line=dict(color='red'),
        marker=dict(color='red')))
    
    fig.add_trace(go.Scatter(
        x=days, 
        y=UBAH_test['values'], 
        mode='lines+markers', 
        name='Buy and Hold',
        line=dict(color='blue'),
        marker=dict(color='blue')))
    
    # Customize the layout
    fig.update_layout(
    title="Portfolio Value Evolution",
    xaxis_title="Days",
    yaxis_title="Portfolio Value",
    legend_title="Strategy"
    )

    # Build portfolio compositon evolution
    portfolio_distribution = [portfolio_weights[i] * asset_memory[i] for i in range(len(asset_memory))]
    portfolio_names = ["Cash", "AAPL", "CMCSA", "CSCO", "FB", "HBAN", "INTC", "MSFT", "MU", "NVDA", "QQQ", "XIV"]
    money_threshold = 1 #threshold of money of the stocks displayed
    portfolio_evolution = []
    for distrib in portfolio_distribution:
        portfolio_daily = dict()
        for j in range(len(distrib)):
            if distrib[j]>money_threshold:
                portfolio_daily[portfolio_names[j]] = distrib[j]
        portfolio_evolution.append(portfolio_daily)

    return fig, portfolio_evolution, days

def main():
    st.set_page_config(page_title="Financial trading copilot", 
                       page_icon="📈", 
                       layout="wide", 
                       initial_sidebar_state="expanded")
    st.title("Financial trading copilot")
    
    initial_amount = st.number_input("Enter an integer:", min_value=1000, max_value=1000000, step=1000, value=10000)
    
    if st.button("Test"):
        with st.spinner("Testing in progress, please wait for the results"):
            graph, portfolio_evolution, days = process(initial_amount)
            st.session_state.graph = graph
            st.session_state.portfolio_evolution = portfolio_evolution
            st.session_state.days = days

    if 'graph' in st.session_state:
        st.plotly_chart(st.session_state.graph)

    if 'days' in st.session_state:
        selected_day = st.selectbox('Select a day:', st.session_state.days)
        if selected_day:
            index = (st.session_state.days == selected_day).argmax()
            portfolio = st.session_state.portfolio_evolution[index]

            st.subheader(f'Portfolio on Day {selected_day}:')  # Using st.subheader

            portfolio_df = pd.DataFrame(list(portfolio.items()), columns=['Stock', 'Value ($)'])
            portfolio_df = portfolio_df.style.set_table_attributes('style="width:auto; min-width:150px; white-space: nowrap;"')
            st.dataframe(portfolio_df, hide_index=True)


if __name__ == "__main__":
    main()
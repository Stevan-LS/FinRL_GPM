import streamlit as st
from test import test
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def process(selected_stocks, initial_amount):
    GPM_test, UBAH_test, portfolio_weights, asset_memory = test(selected_stocks, initial_amount)

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

    return GPM_test, UBAH_test, portfolio_evolution

def build_graph(GPM_values, UBAH_values):
    days = np.arange(1, len(GPM_values)+1)
    # Build figure of portfolio value evolution
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, 
        y=GPM_values, 
        mode='lines+markers', 
        name='GPM',
        line=dict(color='red'),
        marker=dict(color='red')))
    
    fig.add_trace(go.Scatter(
        x=days, 
        y=UBAH_values, 
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

    return fig, days

def get_list_of_stocks():
    nasdaq_temporal = pd.read_csv("Temporal_Relational_Stock_Ranking_FinRL/temporal_data/NASDAQ_temporal_data.csv")
    return nasdaq_temporal["tic"].unique().tolist()

def main():
    st.set_page_config(page_title="Financial trading copilot", 
                       page_icon="📈", 
                       layout="wide", 
                       initial_sidebar_state="expanded")
    st.title("Financial trading copilot")
    
    #Define the initial amount to invest in the portfolio
    initial_amount = st.number_input("Enter your initial investment amount:", min_value=1000, max_value=1000000, step=1000, value=10000)
    
    #Select the companies you want to trade the stocks (companies that can possibly be in you portfolio)
    stocks = get_list_of_stocks()

    # Multiselect for company selection
    selected_stocks = st.multiselect("Select companies", options=stocks, default=["AAPL", "CMCSA", "CSCO", "FB", "HBAN", "INTC", "MSFT", "MU", "NVDA", "QQQ", "XIV"])
    
    if st.button("Test"):
        with st.spinner("Backtesting in progress, please wait for the results"):
            GPM_test, UBAH_test, portfolio_evolution = process(selected_stocks, initial_amount)

            st.session_state.portfolio_evolution = portfolio_evolution
            st.session_state.GPM_test = GPM_test
            st.session_state.UBAH_test = UBAH_test
    
    if 'GPM_test' in st.session_state:
        timeframe = st.selectbox("Select a timeframe:", ["Overall", "Monthly", "Weekly"])
        if timeframe == "Overall":
            GPM_values = st.session_state.GPM_test['values']
            UBAH_values = st.session_state.UBAH_test['values']
            st.session_state.portfolio_evol_timeframe = st.session_state.portfolio_evolution
        elif timeframe == "Monthly":
            if len(st.session_state.GPM_test['values'])>=30:
                GPM_values = st.session_state.GPM_test['values'][-30:]
                UBAH_values = st.session_state.UBAH_test['values'][-30:]
                st.session_state.portfolio_evol_timeframe = st.session_state.portfolio_evolution[-30:]
            else:
                GPM_values = st.session_state.GPM_test['values']
                UBAH_values = st.session_state.UBAH_test['values']
                st.session_state.portfolio_evol_timeframe = st.session_state.portfolio_evolution
        elif timeframe == "Weekly":
            if len(st.session_state.GPM_test['values'])>=7:
                GPM_values = st.session_state.GPM_test['values'][-7:]
                UBAH_values = st.session_state.UBAH_test['values'][-7:]
                st.session_state.portfolio_evol_timeframe = st.session_state.portfolio_evolution[-7:]
            else:
                GPM_values = st.session_state.GPM_test['values']
                UBAH_values = st.session_state.UBAH_test['values']
                st.session_state.portfolio_evol_timeframe = st.session_state.portfolio_evolution

        timeframe_return = (GPM_values[-1]/GPM_values[-0] - 1)*100
        st.metric(label=f"{timeframe} return", value=f"{timeframe_return:.4g}%")
        graph, days = build_graph(GPM_values, UBAH_values)
        st.session_state.graph = graph
        st.session_state.days = days
        st.plotly_chart(st.session_state.graph)

    if 'days' in st.session_state:
        selected_day = st.selectbox('Select a day:', st.session_state.days)
        if selected_day:
            index = (st.session_state.days == selected_day).argmax()

            st.subheader(f'Portfolio description on day {selected_day}:')

            portfolio = st.session_state.portfolio_evol_timeframe[index]
            
            portfolio_df = pd.DataFrame(list(portfolio.items()), columns=['Stock', 'Value ($)'])

            col1, col2 = st.columns([0.5, 2])

            with col1:
                styled_portfolio_df = portfolio_df.style.format({"Value ($)": "{:.2f}"}).set_table_attributes('style="width:100%; min-width:200px; white-space: nowrap;"')
                st.dataframe(styled_portfolio_df, hide_index=True)

                if index > 0:
                    previous_portfolio = st.session_state.portfolio_evol_timeframe[index-1]
                    
                    # Convert previous and current portfolios to DataFrames
                    previous_df = pd.DataFrame(list(previous_portfolio.items()), columns=['Stock', 'Previous Value ($)'])
                    merged_df = pd.merge(portfolio_df, previous_df, on='Stock', how='outer').fillna(0)
                    
                    # Calculate the difference
                    merged_df['Amount ($)'] = merged_df['Value ($)'] - merged_df['Previous Value ($)']

                    # Separate into bought and sold stocks
                    bought_df = merged_df[merged_df['Amount ($)'] > 0]
                    sold_df = merged_df[merged_df['Amount ($)'] < 0]
                    
                    # Display bought stocks
                    if not bought_df.empty:
                        st.markdown("**Bought Stocks:**")
                        bought_df = bought_df[bought_df['Stock'] != 'Cash'][['Stock', 'Amount ($)']].style.format({"Change ($)": "{:.2f}"})
                        st.dataframe(bought_df, hide_index=True)
                    
                    # Display sold stocks
                    if not sold_df.empty:
                        st.markdown("**Sold Stocks:**")
                        sold_df = sold_df[sold_df['Stock'] != 'Cash'][['Stock', 'Amount ($)']].style.format({"Change ($)": "{:.2f}"})
                        st.dataframe(sold_df, hide_index=True)

            with col2:
                # Use the unstyled DataFrame for the pie chart
                fig = px.pie(portfolio_df, names='Stock', values='Value ($)',
                             hole=0.3,  # Creates a donut chart (optional)
                             color_discrete_sequence=px.colors.sequential.RdBu)
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=True)

                st.plotly_chart(fig)


if __name__ == "__main__":
    main()
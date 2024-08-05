import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import GPM
import time

def test(initial_amount=10000):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    nasdaq_temporal = pd.read_csv("Temporal_Relational_Stock_Ranking_FinRL/temporal_data/NASDAQ_temporal_data.csv")
    nasdaq_edge_index = np.load("Temporal_Relational_Stock_Ranking_FinRL/relational_data/edge_indexes/NASDAQ_sector_industry_edge_index.npy")
    nasdaq_edge_type = np.load("Temporal_Relational_Stock_Ranking_FinRL/relational_data/edge_types/NASDAQ_sector_industry_edge_type.npy")

    list_of_stocks = nasdaq_temporal["tic"].unique().tolist()
    tics_in_portfolio = ["AAPL", "CMCSA", "CSCO", "FB", "HBAN", "INTC", "MSFT", "MU", "NVDA", "QQQ", "XIV"]

    portfolio_nodes = []
    for tic in tics_in_portfolio:
        portfolio_nodes.append(list_of_stocks.index(tic))
    portfolio_nodes

    nodes_kept, new_edge_index, nodes_to_select, edge_mask = k_hop_subgraph(
        torch.LongTensor(portfolio_nodes),
        2,
        torch.from_numpy(nasdaq_edge_index),
        relabel_nodes=True,
    )

    # reduce temporal data
    nodes_kept = nodes_kept.tolist()
    nasdaq_temporal["tic_id"], _ = pd.factorize(nasdaq_temporal["tic"], sort=True)
    nasdaq_temporal = nasdaq_temporal[nasdaq_temporal["tic_id"].isin(nodes_kept)]
    nasdaq_temporal = nasdaq_temporal.drop(columns="tic_id")
    nasdaq_temporal

    # reduce edge type
    new_edge_type = torch.from_numpy(nasdaq_edge_type)[edge_mask]
    _, new_edge_type = torch.unique(new_edge_type, return_inverse=True)
    new_edge_type

    df_portfolio = nasdaq_temporal[["day", "tic", "close", "high", "low"]]
    df_portfolio_train = df_portfolio[df_portfolio["day"] < 979]
    df_portfolio_test = df_portfolio[df_portfolio["day"] >= 1180]
    environment_train = PortfolioOptimizationEnv(
            df_portfolio_train,
            initial_amount=initial_amount,
            comission_fee_pct=0.0025,
            time_window=50,
            features=["close", "high", "low"],
            time_column="day",
            normalize_df=None, # dataframe is already normalized
            tics_in_portfolio=tics_in_portfolio
        )
    environment_test = PortfolioOptimizationEnv(
                df_portfolio_test,
                initial_amount=initial_amount,
                comission_fee_pct=0.0025,
                time_window=50,
                features=["close", "high", "low"],
                time_column="day",
                normalize_df=None, # dataframe is already normalized
                tics_in_portfolio=tics_in_portfolio
            )
    
    # set PolicyGradient parameters
    model_kwargs = {
        "lr": 0.01,
        "policy": GPM,
    }
    # here, we can set GPM's parameters
    policy_kwargs = {
        "edge_index": new_edge_index,
        "edge_type": new_edge_type,
        "nodes_to_select": nodes_to_select
    }
    model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)

    GPM_test = {
        "values": {},
    }

    # instantiate an architecture with the same arguments used in training
    # and load with load_state_dict.
    policy = GPM(new_edge_index, new_edge_type, nodes_to_select, device=device)
    policy.load_state_dict(torch.load("policy_GPM.pt"))

    # testing
    DRLAgent.DRL_validation(model, environment_test, policy=policy)
    GPM_test["values"] = environment_test._asset_memory["final"]
    #GPM_test["days"] = df_portfolio_test['day']

    portfolio_weights = environment_test._final_weights
    asset_memory = environment_test._asset_memory['final']


    UBAH_test = {
        "values": {},
    }

    PORTFOLIO_SIZE = len(tics_in_portfolio)

    # test period
    terminated = False
    environment_test.reset()
    while not terminated:
        action = [0] + [1/PORTFOLIO_SIZE] * PORTFOLIO_SIZE
        _, _, terminated, _ = environment_test.step(action)
    UBAH_test["values"] = environment_test._asset_memory["final"]
    #UBAH_test["days"] = df_portfolio_test['day']

    return GPM_test, UBAH_test, portfolio_weights, asset_memory

if __name__ == '__main__':
    start_time = time.time()
    test()
    duration = time.time()-start_time
    print("test duration", duration)
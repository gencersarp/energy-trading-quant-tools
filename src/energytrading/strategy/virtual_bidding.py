import pandas as pd
import numpy as np


def virtual_bidding_pnl(
    da_price: pd.Series, 
    rt_price: pd.Series, 
    bid_type: pd.Series, 
    mw_volume: pd.Series,
    transaction_fee: float = 0.05
) -> pd.Series:
    """
    Simulates Virtual Bidding (Convergence Bidding) PnL in US ISO markets (ERCOT, PJM, CAISO).
    
    bid_type: 
        1  -> INC (Increment): Sell DA, Buy RT. Profit = DA - RT
        -1 -> DEC (Decrement): Buy DA, Sell RT. Profit = RT - DA
        0  -> No bid
    
    transaction_fee: Cleared MWh fee imposed by the ISO.
    """
    # Align indices
    df = pd.DataFrame({"DA": da_price, "RT": rt_price, "Type": bid_type, "Vol": mw_volume}).dropna()
    
    pnl = np.zeros(len(df))
    
    # INC Logic
    inc_mask = df["Type"] == 1
    pnl[inc_mask] = df.loc[inc_mask, "Vol"] * (df.loc[inc_mask, "DA"] - df.loc[inc_mask, "RT"])
    
    # DEC Logic
    dec_mask = df["Type"] == -1
    pnl[dec_mask] = df.loc[dec_mask, "Vol"] * (df.loc[dec_mask, "RT"] - df.loc[dec_mask, "DA"])
    
    # Subtract explicit ISO transaction fees per MWh cleared
    active_bids = df["Type"] != 0
    pnl[active_bids] -= (df.loc[active_bids, "Vol"] * transaction_fee)
    
    return pd.Series(pnl, index=df.index)


def moving_average_convergence_signal(da_price: pd.Series, rt_price: pd.Series, window: int = 7) -> pd.Series:
    """
    Generates an INC/DEC signal based on the historical premium between Day-Ahead and Real-Time.
    If DA has been historically pricing higher than RT, we sell DA and buy RT (INC).
    """
    spread = da_price - rt_price
    roll_mean = spread.rolling(window).mean()
    
    signal = pd.Series(0, index=da_price.index)
    signal[roll_mean > 0] = 1   # DA is overpriced, submit INC
    signal[roll_mean < 0] = -1  # RT is overpriced, submit DEC
    
    # Shift by 1 day to execute tomorrow based on today's moving average
    return signal.shift(1).fillna(0)
import pandas as pd


def renewable_ramp_signal(
    solar_forecast: pd.Series, 
    wind_forecast: pd.Series, 
    ramp_threshold_mw: float = 1000.0
) -> pd.Series:
    """
    Generates trading signals based on the 'Duck Curve' and sudden renewable drop-offs.
    If renewable generation drops faster than the threshold, it triggers a LONG signal
    anticipating a price spike as expensive gas peaker plants are activated.
    """
    total_renewables = solar_forecast + wind_forecast
    
    # 1-period difference (ramp rate)
    ramp = total_renewables.diff()
    
    signal = pd.Series(0, index=total_renewables.index)
    # Massive drop in renewables -> Buy Power (Spike incoming)
    signal[ramp < -ramp_threshold_mw] = 1
    # Massive surge in renewables -> Sell Power (Negative pricing risk)
    signal[ramp > ramp_threshold_mw] = -1
    
    return signal
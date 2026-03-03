import pandas as pd


def calculate_clean_spark_spread(
    power_price: pd.Series, 
    gas_price: pd.Series, 
    carbon_price: pd.Series, 
    efficiency: float = 0.50, 
    emission_factor: float = 0.2  # tons of CO2 per MWh of gas thermal energy
) -> pd.Series:
    """
    Calculates the Clean Spark Spread (CSS), representing the theoretical profit margin
    of a gas-fired power plant.
    
    CSS = Power - (Gas / Efficiency) - (Carbon * Emission Factor / Efficiency)
    Note: Emission factor must be adjusted by efficiency to get emissions per MWh of electrical output.
    """
    gas_cost = gas_price / efficiency
    carbon_cost = carbon_price * (emission_factor / efficiency)
    
    return power_price - gas_cost - carbon_cost


def calculate_clean_dark_spread(
    power_price: pd.Series, 
    coal_price: pd.Series, 
    carbon_price: pd.Series, 
    efficiency: float = 0.35, 
    emission_factor: float = 0.34  # tons of CO2 per MWh of coal thermal energy
) -> pd.Series:
    """
    Calculates the Clean Dark Spread (CDS) for coal-fired power plants.
    Coal plants have lower efficiency and higher emission factors than gas plants.
    """
    coal_cost = coal_price / efficiency
    carbon_cost = carbon_price * (emission_factor / efficiency)
    
    return power_price - coal_cost - carbon_cost
from datetime import datetime, timedelta
import pandas as pd

def filter_by_time(df: pd.DataFrame, date_column: str, period: str = "today") -> pd.DataFrame:
    """
    Filters a DataFrame based on a time period.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to filter.
    - date_column (str): Column name with datetime values.
    - period (str): "today", "week", "month", or "custom".
    
    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")
    
    df[date_column] = pd.to_datetime(df[date_column])
    now = datetime.now()

    if period == "today":
        filtered = df[df[date_column].dt.date == now.date()]
    elif period == "week":
        start = now - timedelta(days=7)
        filtered = df[df[date_column] >= start]
    elif period == "month":
        start = now - timedelta(days=30)
        filtered = df[df[date_column] >= start]
    else:
        raise ValueError(f"Unsupported period: {period}")
    
    return filtered
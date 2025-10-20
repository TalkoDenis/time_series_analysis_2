import pandas as pd

from prophet import Prophet
import plotly.graph_objects as go


def read_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")

    except pd.errors.EmptyDataError:
        print(f"Error: The file '{path}' is empty or malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    else:
        return df

def validate_data(df):
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    except Exception as e:
        print(f"Failed to convert to datetime: {e}")


def validate_data(df):
    from pandas.errors import OutOfBoundsDatetime
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], downcast='integer')
    except ValueError as e:
        print("Some values couldn’t be parsed as dates.")
        print(e)
    except OutOfBoundsDatetime as e:
        print("Date values out of range (too large/small). Setting invalid to NaT.")
        print(e)
    except TypeError as e:
        print("Invalid type for date column (maybe dicts or lists?).")
        print(e)
    else:
        return df.sort_values(by=df.columns[0])


def rename_columns(df):
    df_prophet = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
    return df_prophet

def get_min_data(df):
    return df['ds'].min()

def get_max_data(df):
    return df['ds'].max()

def split_df(df_prophet, data='2025-01-01'):
    if pd.to_datetime(data) < get_min_data(df_prophet):
        raise Exception(f'{data} is not valid! It is too small!')
    if pd.to_datetime(data) > get_max_data(df_prophet):
        raise Exception(f'{data} is not valid! It is too big!')
    train_df = df_prophet[df_prophet['ds'] <= pd.to_datetime(data)]
    future_df = df_prophet[df_prophet['ds'] > pd.to_datetime(data)]
    return train_df, future_df


def model_learning(train_df, future_df, seasonality_prior_scale=25.0, country_name='US'):
    if len(country_name) < 2:
        raise Exception(f'{country_name} is too short!')
    model = Prophet(
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_country_holidays(country_name=country_name)
    model.fit(train_df)

    prediction_dates = future_df[['ds']]
    forecast = model.predict(prediction_dates)
    return forecast

def visualize_data(df_prophet, forecast):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_prophet['ds'],
        y=df_prophet['y'],
        mode='lines',
        line=dict(color='red', width=2),
        name='Now'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Forecast'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        name='Forecast interval'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        showlegend=False
    ))

    fig.update_layout(
        title='Our data',
        xaxis_title='Week',
        yaxis_title='N',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()

def save_result_df(future_df, forecast):
    results_df_org = pd.merge(future_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    results_df_org.rename(columns={
        'ds': 'week',
        'y': 'fd_cnt',
        'yhat': 'forecast',
        'yhat_lower': 'min_forecast',
        'yhat_upper': 'max_forecast'
    }, inplace=True)
    return results_df_org




df = read_data('./data/data.csv')
df = validate_data(df)
df_prophet = rename_columns(df)
del df
train_df, future_df = split_df(df_prophet)
forecast = model_learning(train_df, future_df, seasonality_prior_scale=25.0, country_name='US')
visualize_data(df_prophet, forecast)
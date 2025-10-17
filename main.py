import pandas as pd
# import matplotlib.pyplot as plt

import clickhouse_connect
import os

from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import io




df = df_org_metric.copy()

# --- 1. Подготовка данных ---
# Гарантируем, что колонка 'week' имеет формат даты
df['week'] = pd.to_datetime(df['week'])

# Prophet требует, чтобы столбцы назывались 'ds' (дата) и 'y' (значение)
df_prophet = df.rename(columns={'week': 'ds', 'fd_cnt': 'y'})


# --- 2. Разделение данных ---
# Данные для обучения модели (до 2025-05-01 включительно)
train_df = df_prophet[df_prophet['ds'] <= '2025-05-01']

# Данные, для которых нужно построить прогноз (после 2025-05-01)
future_df = df_prophet[df_prophet['ds'] > '2025-05-01']


# --- 3. Обучение модели ---
# Инициализируем и обучаем модель
model = Prophet(
    seasonality_prior_scale=25.0 
)

model.add_country_holidays(country_name='AR') # Аргентина
model.add_country_holidays(country_name='VE') # Венесуэла
model.add_country_holidays(country_name='CL') # Чили
model.add_country_holidays(country_name='MX') # Мексика
model.add_country_holidays(country_name='PE') # Перу

model.fit(train_df)


# --- 4. Создание прогноза ---
# Для предсказания Prophet'у нужен датафрейм только с колонкой 'ds'
prediction_dates = future_df[['ds']]
forecast = model.predict(prediction_dates)


# --- 5. Интерактивная визуализация (Plotly) ---
# print("\nСоздание интерактивного графика...")
fig = go.Figure()

# --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
# Добавляем фактические данные за ВЕСЬ ПЕРИОД в виде линии
fig.add_trace(go.Scatter(
    x=df_prophet['ds'],  # Используем полный датафрейм df_prophet
    y=df_prophet['y'],
    mode='lines',
    line=dict(color='red', width=2),
    name='Факт'
))

# Добавляем линию прогноза
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    line=dict(color='blue', width=2),
    name='Прогноз'
))

# Добавляем закрашенную область прогнозного интервала
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0,100,80,0.2)',
    name='Прогнозный интервал'
))
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0,100,80,0.2)',
    showlegend=False # Скрываем легенду для нижней границы
))

# Добавляем вертикальную линию, разделяющую историю и прогноз
fig.add_vline(x=pd.to_datetime('2025-05-01'), line_width=2, line_dash="dash", line_color="grey")

# Настраиваем внешний вид графика
fig.update_layout(
    title='Прогноз показателя FD vs Факт (Органика)',
    xaxis_title='Неделя',
    yaxis_title='Показатель FD',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()


# --- 6. Сохранение результата в итоговый датафрейм ---
# Эта часть остается без изменений
results_df_org = pd.merge(future_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

results_df_org.rename(columns={
    'ds': 'week',
    'y': 'fd_cnt',
    'yhat': 'прогноз',
    'yhat_lower': 'минимальный_прогнозный_интервал',
    'yhat_upper': 'максимальный_прогнозный_интервал'
}, inplace=True)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Заголовок приложения
st.title("Прогноз цен на арматуру")
st.write("Укажите дату понедельника недели, для которой сделать прогноз:")

# Загрузка модели
@st.cache_resource
def load_model():
    with open('catboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Функция для создания фичи из даты
def create_date_features(date):
    return {
        'year': date.year,
        'month': date.month,
        'week_of_year': date.isocalendar().week,
        'quarter': (date.month - 1) // 3 + 1
    }

# Ввод даты
selected_date = st.date_input(
    "Выберите дату понедельника:",
    min_value=datetime.today(),
    help="Укажите любой понедельник будущей недели"
)

# Проверка что выбрана дата понедельника
if selected_date.weekday() != 0:
    st.error("Пожалуйста, выберите дату понедельника!")
else:
    # Создание фичей для модели
    features = create_date_features(selected_date)
    features_df = pd.DataFrame([features])
    
    # Прогноз
    prediction = model.predict(features_df)
    
    # Отображение прогноза
    st.success(f"Прогнозируемая цена на {selected_date.strftime('%d.%m.%Y')}: {prediction[0]:.2f} руб/тонна")
    
    # Генерация данных для графика (пример)
    dates = [selected_date - timedelta(weeks=i) for i in range(4, 0, -1)]
    historical_data = [1200, 1250, 1230, 1270]  # Здесь должны быть реальные исторические данные
    
    # Добавляем прогноз
    dates.append(selected_date)
    historical_data.append(prediction[0])
    
    # Отрисовка графика
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates[-5:], historical_data[-5:], marker='o', linestyle='--', color='b')
    ax.scatter(selected_date, prediction[0], color='r', s=100, label='Прогноз')
    ax.set_title("Динамика цен на арматуру")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена (руб/тонна)")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

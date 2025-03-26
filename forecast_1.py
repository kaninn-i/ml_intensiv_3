# импорт библиотек
import streamlit as st
import pandas as pd
import pickle
from datetime import timedelta
import matplotlib.pyplot as plt

# загрузка модели
with open('lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# загрузка исторических данных
historical_df = pd.read_csv('historical_data.csv')
historical_df['dt'] = pd.to_datetime(historical_df['dt'])
historical_df = historical_df.set_index('dt')

def generate_predictions(start_date, weeks_ahead=6):
    current_date = start_date
    predictions = []
    
    # создаем копию исторических данных и сортируем индекс
    extended_df = historical_df.copy().sort_index()
    
    # прогнозируем на 6 недель вперед
    for week in range(weeks_ahead):
        forecast_date = current_date + timedelta(weeks=week)
        
        # пропускаем даты, которые уже есть в исторических данных
        if forecast_date in extended_df.index:
            continue
            
        # создаем признаки
        features = {
            'year': forecast_date.year,
            'month': forecast_date.month,
            'day': forecast_date.day,
            'week': forecast_date.isocalendar().week,
        }
        
        # рассчитываем лаги с использованием ближайших доступных данных
        for lag in [1, 2]:
            lag_date = forecast_date - timedelta(weeks=lag)
            available_dates = extended_df.index[extended_df.index <= lag_date]
            
            if len(available_dates) == 0:
                st.error(f"Нет данных для расчета лага {lag} на дату {forecast_date.strftime('%Y-%m-%d')}")
                return None
                
            closest_date = available_dates.max()
            features[f'lag_{lag}'] = extended_df.loc[closest_date, 'Цена на арматуру']
        
        # прогнозируем цену
        prediction = model.predict(pd.DataFrame([features]))[0]
        
        # добавляем прогноз в датафрейм
        extended_df.loc[forecast_date] = {
            'Цена на арматуру': prediction,
            'lag_1': features['lag_1'],
            'lag_2': features['lag_2']
        }
        
        predictions.append((forecast_date, prediction))
    
    return pd.DataFrame(predictions, columns=['Date', 'Predicted Price'])

def main():
    st.title('Прогнозирование цен на арматуру')
    
    # выбор даты
    last_historical_date = historical_df.index.max().date()
    min_date = last_historical_date + timedelta(weeks=1)
    
    input_date = st.date_input(
        'Выберите начальную дату прогноза (понедельник):',
        min_value=min_date,
        max_value=last_historical_date + timedelta(weeks=26), # ставим максимальное ограничение по дате в полгода (во избежании слишком неверных прогнозов на очень дальние временные рамки)
        value=min_date
    )
    
    if input_date.weekday() != 0:
        st.error("Тендеры проводятся только по понедельникам!")
        return
    
    if st.button('Сформировать прогноз'):
        start_date = pd.to_datetime(input_date)
        
        # прогнозируем
        predictions = generate_predictions(start_date)
        
        if predictions is not None:
            # фильтруем только прогнозируемый период
            forecast_df = predictions.set_index('Date')
            
            # получаем текущую цену
            predicted_prices = forecast_df['Predicted Price'].tolist()
            current_price = predicted_prices[0]
            
            # определяем рекомендацию
            possible_n = 0
            for n in predicted_prices:
                if current_price <= n:
                    possible_n += 1
                elif current_price > n:
                    break

            max_n = possible_n if possible_n != 0 else 1
            
            # визуализация
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(forecast_df.index, forecast_df['Predicted Price'], marker='o', 
                    linestyle='-', linewidth=2, markersize=8, label='Прогнозная цена')
            ax.axhline(y=current_price, color='red', linestyle='--', 
                      label=f'Текущая цена: {current_price:.2f} руб/т')
            ax.set_xlabel('Дата', fontsize=12)
            ax.set_ylabel('Цена (руб/т)', fontsize=12)
            ax.set_title('Прогноз цен на арматуру', fontsize=14, pad=20)
            ax.legend(prop={'size': 10})
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            
            # вывод дополнительных аналитических данных
            st.subheader("Детали прогноза:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Даты и цены:**")
                st.dataframe(forecast_df.reset_index().style.format({
                    'Date': lambda x: x.strftime('%Y-%m-%d'),
                    'Predicted Price': '{:.2f} руб'
                }))
            
            with col2:
                st.write("**Статистика:**")
                st.metric("Средняя цена", f"{forecast_df.mean().values[0]:.2f} руб")
                st.metric("Минимальная цена", f"{forecast_df.min().values[0]:.2f} руб")
                st.metric("Максимальная цена", f"{forecast_df.max().values[0]:.2f} руб")
            
            # вывод рекомендаций
            st.subheader("Рекомендации:")
            with st.container():
                if max_n == 1:
                    st.success(f"Оптимальный период закупки: {max_n} неделя")
                elif max_n in [2, 3, 4]:
                    st.success(f"Оптимальный период закупки: {max_n} недели")
                elif max_n in [5, 6]:
                    st.success(f"Оптимальный период закупки: {max_n} недель")



main()
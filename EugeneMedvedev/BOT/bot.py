from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackContext, filters
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import io

# Токен бота
TOKEN = '7820446476:AAEt5D1b6jo2_MArFEsOsGE71x18Aqoqygg'

# Загрузка модели
model = joblib.load("model/lgbm_model.pkl")

# Функция для генерации признаков из даты
def generate_features(date):
    date = pd.to_datetime(date, format="%d.%m.%Y")
    features = {
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'week': date.isocalendar().week,
        'lag_1': 45000,
        'lag_2': 45000
    }
    return pd.DataFrame([features])

# Проверка, является ли дата понедельником
def is_monday(date):
    return pd.to_datetime(date, format="%d.%m.%Y").weekday() == 0

# Функция прогноза на 6 недель вперед с графиком
def forecast_6_weeks(start_date):
    dates = []
    prices = []
    for i in range(6):
        next_date = pd.to_datetime(start_date) + pd.DateOffset(weeks=i)
        features = generate_features(next_date.strftime("%d.%m.%Y"))
        price = model.predict(features)[0]
        dates.append(next_date)
        prices.append(price)
    return dates, prices

# Генерация графика прогноза
def generate_forecast_plot(dates, prices):
    plt.figure(figsize=(8, 4))
    plt.plot(dates, prices, marker='o', linestyle='-', color='blue', label='Прогноз цены')
    for i, price in enumerate(prices):
        plt.text(dates[i], price, f'{price:.2f}', ha='center', va='bottom')
    plt.title('Прогноз цены на 6 недель вперёд')
    plt.xlabel('Дата')
    plt.ylabel('Цена на арматуру')
    plt.xticks(dates, [date.strftime('%d.%m') for date in dates], rotation=45)
    plt.grid(True)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# Определение стратегии закупки с учётом общей тенденции
def determine_strategy(prices):
    weeks_to_buy = 1
    for i in range(1, len(prices)):
        # Если текущая цена выше предыдущей — продолжаем закупку
        if prices[i] > prices[i - 1]:
            weeks_to_buy = i + 1
        # Если цена упала, но потом снова выросла выше начальной — продолжаем закупку
        elif prices[i] >= prices[0]:
            weeks_to_buy = i + 1
        # Если цена падает и не восстанавливается — останавливаемся
        else:
            break
    return weeks_to_buy

# Команда /start
async def start(update: Update, context: CallbackContext) -> None:
    message = (
        "👋 Привет! Это бот для прогнозирования цен на арматуру. 🏗️\n"
        "Введите дату в формате 'ДД.ММ.ГГГГ' и получите прогноз."
    )
    await update.message.reply_text(message)

# Обработка текстового ввода даты
async def predict(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    try:
        if not is_monday(user_input):
            await update.message.reply_text("❌ Ошибка: дата должна быть понедельником!")
            return

        dates, prices = forecast_6_weeks(user_input)
        weeks_to_buy = determine_strategy(prices)
        strategy = f"📦 Закупаем на {weeks_to_buy} недель."
        trend = 'растёт' if weeks_to_buy > 1 else 'падает или стабильна'
        graph = generate_forecast_plot(dates, prices)
        response = (
            f"💰 Прогноз цены на {user_input}: {prices[0]:.2f} руб.\n"
            f"📈 {strategy}\n"
            f"📊 Рекомендация: цена {trend}."
        )

        await update.message.reply_photo(photo=graph, caption=response)
    except Exception as e:
        response = f"❌ Ошибка: {str(e)}"
        await update.message.reply_text(response)

# Основная функция запуска бота
def main():
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, predict))
    print("✅ Бот запущен!")
    application.run_polling()

if __name__ == '__main__':
    main()

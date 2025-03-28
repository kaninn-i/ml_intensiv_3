# Прогнозирование рыночных цен на арматуру  
**Цель задания:** Создание модели для прогнозирования цен на арматуру и разработка пользовательского интерфейса для рекомендаций по объёму тендера.

**Участники**: Канин Илья, Новасельский Артём, Медведев Евгений
---

## Этапы реализации

### 1. Предобработка данных  
#### Объединение и очистка данных:
- **Соединение данных:** Объединение исторических данных о ценах на арматуру (2015–2023 гг.) из файлов `train.xlsx` и `test.xlsx` в единый датасет
- **Интеграция дополнительных данных:** Включены данные из внешних источников (грузоперевозки, индекс LME, цены на сырье и др.). Отклонены малополезные данные (акции, макропоказатели) из-за высокой зашумлённости.  
- **Обработка пропусков:**
  - Интерполяция недостающих значений (линейная и полиномиальная).  
  - Заполнение на основе соседних наблюдений для временных рядов.

---

### 2. Анализ данных  
#### Исследование временного ряда:
- **Декомпозиция ряда:** Выделение компонент тренда, сезонности и остатков с помощью `seasonal_decompose`.  
- **Визуализация:** Построение графиков:  
  - Общий тренд цен.  
  - Сезонные колебания.  
  - Остаточный шум.  
- **Тест Дики-Фуллера:** Проверка стационарности ряда. Результаты показали нестационарность данных, что подтвердило необходимость дифференцирования для моделей ARIMA.  

---

### 3. Построение моделей  
#### Выбор алгоритмов:
- **Градиентный бустинг (CatBoost, LightGBM):**  
  - Причины выбора: Эффективность на табличных данных, устойчивость к переобучению, интерпретируемость.  
  - Использованы расширенные данные (история цен + внешние признаки).  
- **SARIMA:**  
  - Причины выбора: Учёт сезонности и авторегрессии, работа только с историей цен и датами.  
- **Нейросетевые подходы (отклонены):**  
  - Ограничение: Недостаточный объём данных для обучения глубоких архитектур.  
  - Сложность интерпретации результатов для бизнес-задачи.  

#### Сравнение моделей:
| Модель               | Тип модели          | Используемые данные                | MAE      | MAPE    | R²       |
|----------------------|---------------------|------------------------------------|----------|---------|----------|
| CatBoost             | Градиентный бустинг | Дата + Цена + Доп. признаки        | 3389.43  | 0.07%   | 0.7971   |
| CatBoost             | Градиентный бустинг | Только Дата + Цена                 | 2965.28  | 0.05%   | 0.8788   |
| LightGBM             | Градиентный бустинг | Только Дата + Цена                 | 2374.04  | 0.05%   | 0.9106   |
| LightGBM             | Градиентный бустинг | Дата + Цена + Доп. признаки        | 4242.59  | 0.08%   | 0.7374   |
| SARIMA (ARIMA/SARIMAX)| Статистическая      | Только Дата + Цена                | 2073.61  | N/A*    | -0.0291  |

*Для SARIMA получен аномальный MAPE (>1e+16%) из-за нестационарности ряда.   

---

### 4. Разработка приложений  
#### Варианты интерфейсов:
1. **Telegram-бот:**  
2. **Web-интерфейс (Streamlit):**  

---

### 5. Итоги  
- **Цель достигнута:** Реализованы модели, обеспечивающие прогноз цен на арматуру с возможностью интеграции в приложение.  
- **Преимущества решения:**  
  - Поддержка двух сценариев: с дополнительными данными и без них.  
  - Интуитивный интерфейс для категорийных менеджеров.  
- **Перспективы:** Расширение набора данных, интеграция с ERP-системами.  

---
*Метрики качества будут добавлены после завершения тестирования.*
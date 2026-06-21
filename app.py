import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ----------------------
# НАСТРОЙКА СТРАНИЦЫ
# ----------------------

st.set_page_config(
    page_title='Оценка успешности стартапов',
    page_icon='📊',
    layout='wide'
)

# ----------------------
# ЗАГОЛОВОК
# ----------------------

st.title('📊 Система оценки успешности стартапов')

st.markdown('''
Данная система использует методы машинного обучения
для прогнозирования вероятности успешности стартапа.
''')


# ----------------------
# ЗАГРУЗКА МОДЕЛИ
# ----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "startup_model.pkl")

model = joblib.load(model_path)
# ----------------------
# ВКЛАДКИ
# ----------------------

prediction_tab, analytics_tab = st.tabs([
    'Предсказание',
    'Аналитика'
])


# =====================================================
# ВКЛАДКА ПРЕДСКАЗАНИЯ
# =====================================================

with prediction_tab:

    st.header('Ввод параметров проекта')

    col1, col2 = st.columns(2)

    with col1:

        market = st.slider(
            'Объем рынка',
            0,
            3,
            2
        )

        differentiation = st.slider(
            'Конкурентная дифференциация',
            0,
            3,
            2
        )

    with col2:

        clients = st.slider(
            'Мнение клиентов',
            0,
            3,
            2
        )

        technology = st.slider(
            'Технологическая готовность',
            0,
            3,
            2
        )


    if st.button('Рассчитать вероятность успеха'):

        input_data = pd.DataFrame({
            'Объем рынка': [market],
            'Конкурентная дифференциация': [differentiation],
            'Мнение клиентов': [clients],
            'Технологическая готовность': [technology]
        })


        probability = model.predict_proba(input_data)[0][1]


        st.subheader('Результат прогнозирования')

        st.metric(
            label='Вероятность успеха',
            value=f'{probability * 100:.1f}%'
        )


        if probability < 0.6:
            st.error('Высокая вероятность неуспеха проекта')

        elif probability < 0.8:
            st.warning('Требуется дополнительный анализ проекта')

        else:
            st.success('Высокая вероятность успешности проекта')


# =====================================================
# ВКЛАДКА АНАЛИТИКИ
# =====================================================
with analytics_tab:

    st.header('Аналитика модели')
    data = pd.read_excel(
    "data/dataset.xlsx")


    data = pd.read_excel(
    "data/dataset.xlsx"
    )

    features = [
    'Объем рынка',
    'Конкурентная дифференциация',
    'Мнение клиентов',
    'Технологическая готовность'
    ]

    X = data[features]
    y = data['Успех']

    pred = model.predict(X)
    proba = model.predict_proba(X)[:,1]


    # ROC CURVE
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)


    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-кривая')

    ax.legend()

    st.pyplot(fig)


    # ВАЖНОСТЬ ПРИЗНАКОВ
    st.subheader('Важность признаков')

    importance_df = pd.DataFrame({
        'Признак': features,
        'Важность': model.feature_importances_
    }).sort_values(by='Важность', ascending=False)

    st.dataframe(importance_df)


    # CONFUSION MATRIX
    st.subheader('Confusion Matrix')

    cm = confusion_matrix(y, pred)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(ax=ax_cm)

    st.pyplot(fig_cm)


    # ТАБЛИЦА ДАННЫХ
    st.subheader('Данные')

    st.dataframe(data)
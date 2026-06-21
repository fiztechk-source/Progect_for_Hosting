import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# загрузка данных
file_path = 'data/dataset.xlsx'
data = pd.read_excel(file_path)


# признаки
features = [
    'Объем рынка',
    'Конкурентная дифференциация',
    'Мнение клиентов',
    'Технологическая готовность'
]


# target
X = data[features]
y = data['Успех']


# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# модель
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)


# обучение
model.fit(X_train, y_train)


# предсказания
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]


# метрики
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
roc_auc = roc_auc_score(y_test, proba)


print('\nМетрики модели:\n')
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1-score: {f1:.3f}')
print(f'ROC-AUC: {roc_auc:.3f}')


# сохранение модели
joblib.dump(model, 'startup_model.pkl')

print('\nМодель успешно сохранена.')
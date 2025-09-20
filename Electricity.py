import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

# ================
# 1. Cargar datos
# ================
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Separar variables
X = train.drop('Electricity_Cost', axis=1)
y = train['Electricity_Cost']

# Columnas categóricas y numéricas
cat_cols = ['Site_Area', 'Structure_Type']
num_cols = [col for col in X.columns if col not in cat_cols]

# =======================
# 2. Preprocesamiento
# =======================
# OneHotEncoder para categóricas, StandardScaler para numéricas
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Aplicar transformaciones a X
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(test.drop('Electricity_Cost', axis=1))

# Dividir train/validación
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# =======================
# 3. Definir modelo Keras
# =======================
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # salida continua
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# =======================
# 4. Entrenar modelo
# =======================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# =======================
# 5. Evaluar
# =======================
y_pred_val = model.predict(X_val).flatten()
mae = mean_absolute_error(y_val, y_pred_val)
rmse = root_mean_squared_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# =======================
# 6. Predecir en test
# =======================
predictions = model.predict(X_test_processed).flatten()

# Guardar resultados
submission = test.copy()
submission['Electricity_Cost'] = predictions
submission.to_csv('electricity_cost_predictions.csv', index=False)
print("Archivo 'electricity_cost_predictions.csv' generado.")

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Pérdida por época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('curve_loss.png', dpi=120)
plt.show()


plt.figure()
plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE por época')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('curve_mae.png', dpi=120)
plt.show()



plt.figure()
plt.scatter(y_val, y_pred_val, s=10)
miny = min(y_val.min(), y_pred_val.min())
maxy = max(y_val.max(), y_pred_val.max())
plt.plot([miny, maxy], [miny, maxy])  # línea ideal y = x
plt.title('Real vs Predicho (validación)')
plt.xlabel('Real')
plt.ylabel('Predicho')
plt.tight_layout()
plt.savefig('scatter_real_vs_pred.png', dpi=120)
plt.show()


residuals = y_val - y_pred_val

plt.figure()
plt.hist(residuals, bins=40)
plt.title('Histograma de residuales (validación)')
plt.xlabel('Residual (y_real - y_pred)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('hist_residuals.png', dpi=120)
plt.show()
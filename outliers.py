import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargamos el archivo CSV
df = pd.read_csv('/Users/sebastianromero/Desktop/valores_atipicos/ventas_totales_sinnulos.csv')

# Seleccionamos tres columnas para análisis de valores atípicos
columns_to_analyze = ['ventas_precios_corrientes', 'ventas_precios_constantes', 'ventas_totales_canal_venta']

# Función para remover outliers utilizando Desviación Estándar
def remove_outliers_std(df, column, z_score_threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df_filtered = df[np.abs(df[column] - mean) <= (z_score_threshold * std)]
    return df_filtered

# Función para remover outliers utilizando Rango Intercuartílico
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Aplicamos la eliminación de outliers para las tres columnas elegidas
for column in columns_to_analyze:
    df = remove_outliers_std(df, column)
    df = remove_outliers_iqr(df, column)

# Creamos los diagramas de caja para las tres columnas seleccionadas después de eliminar outliers
plt.figure(figsize=(15, 5))
for i, column in enumerate(columns_to_analyze, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[column])
    plt.title(f'Diagrama de caja post-eliminación de outliers para {column}')

plt.tight_layout()
plt.show()

# Guardamos los datos limpios en archivos CSV
for column in columns_to_analyze:
    clean_df = df[['indice_tiempo', column]].copy()
    clean_df.to_csv(f'{column}_clean.csv', index=False)




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo Excel en un DataFrame de pandas
gastos_df = pd.read_excel('/Users/sebastianromero/Desktop/valores_atipicos/gastos_costos_20_23.xlsx')

# Definir las columnas numéricas a analizar
numeric_columns = ['IMPORTE', 'IVA', 'TOTAL MX']

# Reemplazar los valores nulos en las columnas numéricas con la media de cada una
for column in numeric_columns:
    gastos_df[column].fillna(gastos_df[column].mean(), inplace=True)

# Función para eliminar outliers utilizando Desviación Estándar
def remove_outliers_std(df, column, z_score_threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df_filtered = df[np.abs(df[column] - mean) <= (z_score_threshold * std)]
    return df_filtered

# Función para eliminar outliers utilizando Rango Intercuartílico
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Aplicar la eliminación de outliers para las columnas numéricas
for column in numeric_columns:
    gastos_df = remove_outliers_std(gastos_df, column)
    gastos_df = remove_outliers_iqr(gastos_df, column)

# Crear los diagramas de caja para las columnas seleccionadas después de la eliminación de outliers
plt.figure(figsize=(18, 6))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(1, len(numeric_columns), i)
    sns.boxplot(y=gastos_df[column])
    plt.title(f'Diagrama de caja para {column} después de eliminar outliers')

plt.tight_layout()
plt.show()

# Guardar los datos procesados en archivos CSV en la carpeta de valores atípicos en el escritorio
for column in numeric_columns:
    cleaned_df = gastos_df[['FECHA', 'FOLIO', 'UUID', 'RFC', 'PROVEEDOR', 'TIPO GASTO', column]].copy()
    cleaned_df.to_csv(f'/Users/sebastianromero/Desktop/valores_atipicos/{column}_cleaned.csv', index=False)


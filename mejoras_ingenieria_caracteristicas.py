import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MejorasIngenieriaCaracteristicas:
    """
    Clase para implementar mejoras en la ingeniería de características
    para detección de anomalías en paneles fotovoltaicos
    """
    
    def __init__(self):
        self.caracteristicas_calculadas = False
    
    def crear_caracteristicas_temporales_avanzadas(self, df):
        """
        Crear características temporales más sofisticadas
        """
        print("🕐 CREANDO CARACTERÍSTICAS TEMPORALES AVANZADAS")
        print("="*60)
        
        df = df.copy()
        
        # ========================================
        # 1. CARACTERÍSTICAS CÍCLICAS
        # ========================================
        
        # Hora del día (ciclo de 24 horas)
        df['hora_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # Día del año (ciclo anual)
        df['dia_ano_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['dia_ano_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        
        # Día de la semana (ciclo semanal)
        df['dia_semana_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # ========================================
        # 2. CARACTERÍSTICAS DE ESTACIONALIDAD
        # ========================================
        
        # Variables dummy para estaciones (solo numéricas)
        df['es_invierno'] = df.index.month.isin([12, 1, 2]).astype(int)
        df['es_verano'] = df.index.month.isin([6, 7, 8]).astype(int)
        df['es_primavera'] = df.index.month.isin([3, 4, 5]).astype(int)
        df['es_otono'] = df.index.month.isin([9, 10, 11]).astype(int)
        
        # ========================================
        # 3. CARACTERÍSTICAS DE TRANSICIÓN
        # ========================================
        
        # Transiciones día/noche
        df['es_amanecer'] = ((df.index.hour >= 6) & (df.index.hour <= 9)).astype(int)
        df['es_atardecer'] = ((df.index.hour >= 16) & (df.index.hour <= 19)).astype(int)
        df['es_mediodia'] = ((df.index.hour >= 11) & (df.index.hour <= 14)).astype(int)
        
        # ========================================
        # 4. CARACTERÍSTICAS DE TENDENCIA
        # ========================================
        
        # Tendencia temporal (días desde inicio)
        df['dias_desde_inicio'] = (df.index - df.index.min()).days
        
        # Tendencia lineal
        df['tendencia_lineal'] = np.arange(len(df))
        
        print(f"✓ Características temporales creadas: {len([col for col in df.columns if 'hora_' in col or 'dia_' in col or 'es_' in col or 'tendencia' in col])} nuevas variables")
        
        return df
    
    def crear_caracteristicas_fisicas_mejoradas(self, df):
        """
        Crear características físicas más sofisticadas
        """
        print("\n⚡ CREANDO CARACTERÍSTICAS FÍSICAS MEJORADAS")
        print("="*55)
        
        # ========================================
        # 1. EFICIENCIA DINÁMICA
        # ========================================
        
        # Eficiencia instantánea
        df['eficiencia_instantanea'] = df['total_ac_power'] / (df['poa_irradiance_o_149574'] + 1e-6)
        
        # Eficiencia normalizada por temperatura
        temp_optima = 25
        df['eficiencia_temp_normalizada'] = df['eficiencia_instantanea'] * (1 + 0.004 * (df['ambient_temperature_o_149575'] - temp_optima))
        
        # ========================================
        # 2. CARACTERÍSTICAS DE POTENCIA
        # ========================================
        
        # Factor de capacidad
        potencia_nominal = df['total_ac_power'].max()  # Asumiendo que el máximo es la potencia nominal
        df['factor_capacidad'] = df['total_ac_power'] / potencia_nominal
        
        # Ratio de potencia DC/AC
        df['ratio_dc_ac'] = df['total_dc_power'] / (df['total_ac_power'] + 1e-6)
        
        # ========================================
        # 3. CARACTERÍSTICAS METEOROLÓGICAS
        # ========================================
        
        # Sensación térmica (wind chill)
        df['sensacion_termica'] = 13.12 + 0.6215 * df['ambient_temperature_o_149575'] - 11.37 * (df['wind_speed_o_149576'] ** 0.16) + 0.3965 * df['ambient_temperature_o_149575'] * (df['wind_speed_o_149576'] ** 0.16)
        
        # Índice de confort térmico para paneles
        df['indice_confort_paneles'] = df['poa_irradiance_o_149574'] / (df['ambient_temperature_o_149575'] + 273.15)
        
        # ========================================
        # 4. CARACTERÍSTICAS DE VARIABILIDAD
        # ========================================
        
        # Coeficiente de variación de irradiancia (ventana móvil)
        df['cv_irradiancia_1h'] = df['poa_irradiance_o_149574'].rolling(window=4).std() / (df['poa_irradiance_o_149574'].rolling(window=4).mean() + 1e-6)
        
        print(f"✓ Características físicas creadas: {len([col for col in df.columns if 'eficiencia' in col or 'factor' in col or 'ratio' in col or 'indice' in col or 'cv_' in col])} nuevas variables")
        
        return df
    
    def crear_caracteristicas_interaccion(self, df):
        """
        Crear características de interacción entre variables
        """
        print("\n🔄 CREANDO CARACTERÍSTICAS DE INTERACCIÓN")
        print("="*50)
        
        # ========================================
        # 1. INTERACCIONES TEMPERATURA-IRRADIANCIA
        # ========================================
        
        df['temp_irradiancia'] = df['ambient_temperature_o_149575'] * df['poa_irradiance_o_149574']
        df['temp_irradiancia_ratio'] = df['ambient_temperature_o_149575'] / (df['poa_irradiance_o_149574'] + 1e-6)
        
        # ========================================
        # 2. INTERACCIONES VIENTO-TEMPERATURA
        # ========================================
        
        df['viento_temp'] = df['wind_speed_o_149576'] * df['ambient_temperature_o_149575']
        df['viento_enfriamiento'] = df['wind_speed_o_149576'] * (df['ambient_temperature_o_149575'] - 25)
        
        # ========================================
        # 3. INTERACCIONES POTENCIA-EFICIENCIA
        # ========================================
        
        df['potencia_eficiencia'] = df['total_ac_power'] * df['eficiencia_instantanea']
        df['potencia_temp'] = df['total_ac_power'] * df['ambient_temperature_o_149575']
        
        # ========================================
        # 4. CARACTERÍSTICAS POLINÓMICAS
        # ========================================
        
        df['irradiancia_cuadrada'] = df['poa_irradiance_o_149574'] ** 2
        df['temperatura_cuadrada'] = df['ambient_temperature_o_149575'] ** 2
        df['viento_cuadrado'] = df['wind_speed_o_149576'] ** 2
        
        print(f"✓ Características de interacción creadas: {len([col for col in df.columns if 'temp_' in col or 'viento_' in col or 'potencia_' in col or 'cuadrada' in col or 'cuadrado' in col])} nuevas variables")
        
        return df
    
    def crear_caracteristicas_lag(self, df, lags=[1, 2, 3, 6, 12, 24]):
        """
        Crear características de lag (valores anteriores)
        """
        print(f"\n⏰ CREANDO CARACTERÍSTICAS DE LAG {lags}")
        print("="*45)
        
        variables_lag = ['total_ac_power', 'poa_irradiance_o_149574', 'ambient_temperature_o_149575', 'eficiencia_instantanea']
        
        for var in variables_lag:
            if var in df.columns:
                for lag in lags:
                    df[f'{var}_lag_{lag}'] = df[var].shift(lag)
                    df[f'{var}_diff_{lag}'] = df[var] - df[var].shift(lag)
                    df[f'{var}_pct_change_{lag}'] = df[var].pct_change(lag)
        
        print(f"✓ Características de lag creadas: {len([col for col in df.columns if 'lag_' in col or 'diff_' in col or 'pct_change_' in col])} nuevas variables")
        
        return df
    
    def crear_caracteristicas_estadisticas_moviles(self, df, ventanas=[4, 12, 24]):
        """
        Crear características estadísticas móviles
        """
        print(f"\n📊 CREANDO CARACTERÍSTICAS ESTADÍSTICAS MÓVILES {ventanas}")
        print("="*60)
        
        variables_estadisticas = ['total_ac_power', 'poa_irradiance_o_149574', 'ambient_temperature_o_149575', 'eficiencia_instantanea']
        
        for var in variables_estadisticas:
            if var in df.columns:
                for ventana in ventanas:
                    # Estadísticas básicas
                    df[f'{var}_mean_{ventana}h'] = df[var].rolling(window=ventana).mean()
                    df[f'{var}_std_{ventana}h'] = df[var].rolling(window=ventana).std()
                    df[f'{var}_min_{ventana}h'] = df[var].rolling(window=ventana).min()
                    df[f'{var}_max_{ventana}h'] = df[var].rolling(window=ventana).max()
                    
                    # Estadísticas avanzadas
                    df[f'{var}_skew_{ventana}h'] = df[var].rolling(window=ventana).skew()
                    df[f'{var}_kurt_{ventana}h'] = df[var].rolling(window=ventana).kurt()
                    
                    # Rangos
                    df[f'{var}_range_{ventana}h'] = df[f'{var}_max_{ventana}h'] - df[f'{var}_min_{ventana}h']
                    df[f'{var}_iqr_{ventana}h'] = df[var].rolling(window=ventana).quantile(0.75) - df[var].rolling(window=ventana).quantile(0.25)
        
        print(f"✓ Características estadísticas móviles creadas: {len([col for col in df.columns if 'mean_' in col or 'std_' in col or 'skew_' in col or 'kurt_' in col or 'range_' in col or 'iqr_' in col])} nuevas variables")
        
        return df
    
    def crear_caracteristicas_anomalia_especificas(self, df):
        """
        Crear características específicas para detección de anomalías
        """
        print("\n🚨 CREANDO CARACTERÍSTICAS ESPECÍFICAS DE ANOMALÍA")
        print("="*60)
        
        # ========================================
        # 1. Z-SCORES DINÁMICOS
        # ========================================
        
        df['zscore_power_24h'] = (df['total_ac_power'] - df['total_ac_power'].rolling(window=24).mean()) / (df['total_ac_power'].rolling(window=24).std() + 1e-6)
        df['zscore_irradiance_24h'] = (df['poa_irradiance_o_149574'] - df['poa_irradiance_o_149574'].rolling(window=24).mean()) / (df['poa_irradiance_o_149574'].rolling(window=24).std() + 1e-6)
        df['zscore_eficiencia_24h'] = (df['eficiencia_instantanea'] - df['eficiencia_instantanea'].rolling(window=24).mean()) / (df['eficiencia_instantanea'].rolling(window=24).std() + 1e-6)
        
        # ========================================
        # 2. OUTLIERS ESTADÍSTICOS
        # ========================================
        
        # Método IQR
        for var in ['total_ac_power', 'poa_irradiance_o_149574', 'eficiencia_instantanea']:
            if var in df.columns:
                Q1 = df[var].rolling(window=24).quantile(0.25)
                Q3 = df[var].rolling(window=24).quantile(0.75)
                IQR = Q3 - Q1
                df[f'{var}_outlier_iqr'] = ((df[var] < (Q1 - 1.5 * IQR)) | (df[var] > (Q3 + 1.5 * IQR))).astype(int)
        
        # ========================================
        # 3. ANOMALÍAS DE CORRELACIÓN
        # ========================================
        
        # Correlación móvil entre irradiancia y potencia
        df['correlacion_irrad_power_12h'] = df['poa_irradiance_o_149574'].rolling(window=12).corr(df['total_ac_power'])
        
        # Desviación de correlación esperada
        correlacion_esperada = 0.95
        df['desviacion_correlacion_12h'] = np.abs(df['correlacion_irrad_power_12h'] - correlacion_esperada)
        
        # ========================================
        # 4. ANOMALÍAS DE PATRÓN
        # ========================================
        
        # Cambios bruscos en eficiencia
        df['cambio_brusco_eficiencia'] = abs(df['eficiencia_instantanea'].diff()) > df['eficiencia_instantanea'].rolling(window=24).std() * 2
        
        # Pérdidas de eficiencia inesperadas
        df['perdida_eficiencia_anomala'] = (df['eficiencia_instantanea'] < df['eficiencia_instantanea'].rolling(window=24).quantile(0.1)).astype(int)
        
        print(f"✓ Características de anomalía creadas: {len([col for col in df.columns if 'zscore_' in col or 'outlier_' in col or 'correlacion_' in col or 'desviacion_' in col or 'cambio_' in col or 'perdida_' in col])} nuevas variables")
        
        return df
    
    def aplicar_todas_mejoras(self, df):
        """
        Aplicar todas las mejoras de ingeniería de características
        """
        print("🚀 APLICANDO TODAS LAS MEJORAS DE INGENIERÍA DE CARACTERÍSTICAS")
        print("="*75)
        
        df_mejorado = df.copy()
        
        # Aplicar mejoras secuencialmente
        df_mejorado = self.crear_caracteristicas_temporales_avanzadas(df_mejorado)
        df_mejorado = self.crear_caracteristicas_fisicas_mejoradas(df_mejorado)
        df_mejorado = self.crear_caracteristicas_interaccion(df_mejorado)
        df_mejorado = self.crear_caracteristicas_lag(df_mejorado)
        df_mejorado = self.crear_caracteristicas_estadisticas_moviles(df_mejorado)
        df_mejorado = self.crear_caracteristicas_anomalia_especificas(df_mejorado)
        
        # Limpiar valores NaN
        df_mejorado = df_mejorado.dropna()
        
        print(f"\n✅ INGENIERÍA DE CARACTERÍSTICAS COMPLETADA")
        print(f"   - Características originales: {len(df.columns)}")
        print(f"   - Características finales: {len(df_mejorado.columns)}")
        print(f"   - Nuevas características: {len(df_mejorado.columns) - len(df.columns)}")
        print(f"   - Registros finales: {len(df_mejorado)}")
        
        self.caracteristicas_calculadas = True
        return df_mejorado
    
    def seleccionar_caracteristicas_importantes(self, df, target='anomalia', metodo='correlacion', top_n=50):
        """
        Seleccionar las características más importantes
        """
        print(f"\n🎯 SELECCIONANDO CARACTERÍSTICAS IMPORTANTES ({metodo}, top {top_n})")
        print("="*65)
        
        # Eliminar la columna 'anomalia' del DataFrame excepto como target
        df = df.copy()
        if target in df.columns:
            features = df.drop(columns=[target])
        else:
            features = df
        
        if metodo == 'correlacion':
            # Selección por correlación
            correlaciones = features.corrwith(df[target]).abs().sort_values(ascending=False)
            caracteristicas_seleccionadas = correlaciones.head(top_n).index.tolist()
            caracteristicas_seleccionadas.append(target)  # Asegurarse de incluir el target
        
        elif metodo == 'mutual_info':
            # Selección por información mutua
            from sklearn.feature_selection import mutual_info_classif
            X = features
            y = df[target]
            mi_scores = mutual_info_classif(X, y)
            caracteristicas_seleccionadas = X.columns[np.argsort(mi_scores)[-top_n:]].tolist()
            caracteristicas_seleccionadas.append(target)
        
        elif metodo == 'lasso':
            # Selección por Lasso
            from sklearn.linear_model import Lasso
            X = features
            y = df[target]
            lasso = Lasso(alpha=0.01)
            lasso.fit(X, y)
            caracteristicas_seleccionadas = X.columns[lasso.coef_ != 0].tolist()
            caracteristicas_seleccionadas.append(target)
        
        print(f"✓ Características seleccionadas: {len(caracteristicas_seleccionadas)}")
        return caracteristicas_seleccionadas

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos de ejemplo
    print("📊 EJEMPLO DE USO DE MEJORAS DE INGENIERÍA DE CARACTERÍSTICAS")
    print("="*70)
    
    # Aquí cargarías tus datos reales
    # df = cargar_tus_datos()
    
    # Aplicar mejoras
    # mejorador = MejorasIngenieriaCaracteristicas()
    # df_mejorado = mejorador.aplicar_todas_mejoras(df)
    
    print("✅ Script de mejoras listo para usar") 
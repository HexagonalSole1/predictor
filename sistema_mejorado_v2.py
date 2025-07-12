# 🔋 Sistema Mejorado V2 - Detección de Anomalías en Paneles Solares
# Versión optimizada con correcciones críticas y mejoras implementadas

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Importar módulos propios
try:
    from mejoras_ingenieria_caracteristicas import MejorasIngenieriaCaracteristicas
    from mejoras_modelo_anomalias import MejorasModeloAnomalias
    print("✅ Módulos propios importados correctamente")
    modulos_disponibles = True
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    modulos_disponibles = False

# Prophet para análisis predictivo
try:
    from prophet import Prophet
    print("✅ Prophet importado correctamente")
    prophet_disponible = True
except ImportError:
    print("⚠️ Prophet no disponible")
    prophet_disponible = False

# Modelos avanzados
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    print("✅ Modelos avanzados importados correctamente")
    modelos_avanzados_disponibles = True
except ImportError as e:
    print(f"⚠️ Error importando modelos avanzados: {e}")
    modelos_avanzados_disponibles = False

class SistemaMejoradoV2:
    """
    Sistema mejorado V2 con correcciones críticas y optimizaciones
    """
    
    def __init__(self):
        self.df_original = None
        self.df_caracteristicas = None
        self.modelos_optimizados = {}
        self.resultados_ml = {}
        self.resultados_prophet = None
        self.score_sistema = 0
        
    def cargar_y_preparar_datos(self, ruta_datos=None):
        """
        Cargar y preparar datos con correcciones implementadas
        """
        print("📊 CARGANDO Y PREPARANDO DATOS - VERSIÓN MEJORADA")
        print("="*65)
        
        # NUEVO: Usar directamente el archivo grande si existe
        archivo_integrado = 'anomalias_detectadas_INTEGRADO.csv'
        if os.path.exists(archivo_integrado):
            print(f"✅ Usando archivo principal: {archivo_integrado}")
            self.df_original = pd.read_csv(archivo_integrado)
            print(f"✓ Datos cargados: {self.df_original.shape}")
            return self.df_original
        
        if ruta_datos and os.path.exists(ruta_datos):
            # Cargar desde archivo específico
            self.df_original = pd.read_csv(ruta_datos)
            print(f"✓ Datos cargados desde: {ruta_datos}")
        else:
            # Cargar datasets combinados (como en el notebook original)
            self.df_original = self._cargar_datasets_combinados()
        
        print(f"✓ Datos cargados: {self.df_original.shape}")
        return self.df_original
    
    def _cargar_datasets_combinados(self):
        """
        Cargar y combinar los 3 datasets como en el notebook original
        """
        # Buscar archivos CSV
        archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
        print(f"Archivos CSV encontrados: {archivos_csv}")
        
        environment_data = None
        irradiance_data = None
        electrical_data = None
        
        # Cargar datasets
        for archivo in archivos_csv:
            if 'environment' in archivo.lower():
                environment_data = pd.read_csv(archivo)
                print(f"✓ Cargado: {archivo}")
            elif 'irradiance' in archivo.lower():
                irradiance_data = pd.read_csv(archivo)
                print(f"✓ Cargado: {archivo}")
            elif 'electrical' in archivo.lower():
                electrical_data = pd.read_csv(archivo)
                print(f"✓ Cargado: {archivo}")
        
        if any(data is None for data in [environment_data, irradiance_data, electrical_data]):
            raise ValueError("No se pudieron cargar todos los datasets necesarios")
        
        # Combinar datasets
        df_combined = self._combinar_datasets(environment_data, irradiance_data, electrical_data)
        return df_combined
    
    def _combinar_datasets(self, environment_data, irradiance_data, electrical_data):
        """
        Combinar los 3 datasets con correcciones implementadas
        """
        # Seleccionar variables del inversor 1
        columnas_inv1 = ['measured_on']
        for col in electrical_data.columns:
            if 'inv_01_' in col:
                columnas_inv1.append(col)
        
        df_inv1 = electrical_data[columnas_inv1].copy()
        
        # Limpiar nombres de columnas
        columnas_limpias = {}
        for columna in df_inv1.columns:
            if columna == 'measured_on':
                columnas_limpias[columna] = columna
            elif columna.startswith('inv_01_'):
                nuevo_nombre = columna.replace('inv_01_', '')
                columnas_limpias[columna] = nuevo_nombre
        
        df_inv1 = df_inv1.rename(columns=columnas_limpias)
        
        # Convertir measured_on a datetime
        environment_data['measured_on'] = pd.to_datetime(environment_data['measured_on'])
        irradiance_data['measured_on'] = pd.to_datetime(irradiance_data['measured_on'])
        df_inv1['measured_on'] = pd.to_datetime(df_inv1['measured_on'])
        
        # Realizar INNER JOIN
        df_combined = environment_data.merge(irradiance_data, on='measured_on', how='inner')
        df_final = df_combined.merge(df_inv1, on='measured_on', how='inner')
        
        print(f"✓ Dataset combinado: {df_final.shape}")
        return df_final
    
    def aplicar_caracteristicas_avanzadas_corregidas(self):
        """
        Aplicar características avanzadas con correcciones implementadas
        """
        print("\n🔧 APLICANDO CARACTERÍSTICAS AVANZADAS CORREGIDAS")
        print("="*65)
        
        df = self.df_original.copy()
        df = df.set_index('measured_on')
        
        # CORRECCIÓN CRÍTICA 1: Crear total_ac_power si no existe
        if 'total_ac_power' not in df.columns:
            ac_power_cols = [col for col in df.columns if 'ac_power' in col]
            if ac_power_cols:
                df['total_ac_power'] = df[ac_power_cols].sum(axis=1)
                print(f"✓ Creado total_ac_power desde {len(ac_power_cols)} columnas")
            else:
                df['total_ac_power'] = df['ac_power']
                print("✓ Usando ac_power como total_ac_power")
        
        # CORRECCIÓN CRÍTICA 2: Crear total_dc_power
        if 'total_dc_power' not in df.columns:
            dc_power_cols = [col for col in df.columns if 'dc_power' in col]
            if dc_power_cols:
                df['total_dc_power'] = df[dc_power_cols].sum(axis=1)
                print(f"✓ Creado total_dc_power desde {len(dc_power_cols)} columnas")
            else:
                # Calcular DC power desde corriente y voltaje
                if 'dc_current' in df.columns and 'dc_voltage' in df.columns:
                    df['total_dc_power'] = df['dc_current'] * df['dc_voltage']
                    print("✓ Calculado total_dc_power desde dc_current * dc_voltage")
                else:
                    df['total_dc_power'] = 0
                    print("⚠️ No se pudo calcular total_dc_power")
        
        # Aplicar características avanzadas si los módulos están disponibles
        if modulos_disponibles:
            try:
                mejorador = MejorasIngenieriaCaracteristicas()
                df_caracteristicas = mejorador.aplicar_todas_mejoras(df)
                print(f"✅ Características avanzadas aplicadas: {df_caracteristicas.shape}")
                self.df_caracteristicas = df_caracteristicas
                return df_caracteristicas
            except Exception as e:
                print(f"⚠️ Error en características avanzadas: {e}")
                print("Aplicando características básicas mejoradas...")
        
        # Características básicas mejoradas como respaldo
        df_caracteristicas = self._aplicar_caracteristicas_basicas_mejoradas(df)
        self.df_caracteristicas = df_caracteristicas
        return df_caracteristicas
    
    def _aplicar_caracteristicas_basicas_mejoradas(self, df):
        """
        Aplicar características básicas mejoradas como respaldo
        """
        print("Aplicando características básicas mejoradas...")
        
        # Características temporales avanzadas
        df['hora_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dia_ano_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['dia_ano_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        
        # Estacionalidad
        df['es_invierno'] = df.index.month.isin([12, 1, 2]).astype(int)
        df['es_verano'] = df.index.month.isin([6, 7, 8]).astype(int)
        df['es_primavera'] = df.index.month.isin([3, 4, 5]).astype(int)
        df['es_otono'] = df.index.month.isin([9, 10, 11]).astype(int)
        
        # Transiciones
        df['es_amanecer'] = ((df.index.hour >= 6) & (df.index.hour <= 9)).astype(int)
        df['es_atardecer'] = ((df.index.hour >= 16) & (df.index.hour <= 19)).astype(int)
        df['es_mediodia'] = ((df.index.hour >= 11) & (df.index.hour <= 14)).astype(int)
        
        # Eficiencia
        if 'total_ac_power' in df.columns and 'poa_irradiance_o_149574' in df.columns:
            df['eficiencia_instantanea'] = df['total_ac_power'] / (df['poa_irradiance_o_149574'] + 1e-6)
        
        # Interacciones
        if 'ambient_temperature_o_149575' in df.columns and 'poa_irradiance_o_149574' in df.columns:
            df['temp_irradiancia'] = df['ambient_temperature_o_149575'] * df['poa_irradiance_o_149574']
        
        # Estadísticas móviles básicas
        if 'total_ac_power' in df.columns:
            df['potencia_mean_4h'] = df['total_ac_power'].rolling(window=4).mean()
            df['potencia_std_4h'] = df['total_ac_power'].rolling(window=4).std()
        
        print(f"✓ Características básicas mejoradas creadas: {df.shape[1]} columnas")
        return df
    
    def crear_etiquetas_anomalia_mejoradas(self):
        """
        Crear etiquetas de anomalía con métodos mejorados
        """
        print("\n🏷️ CREANDO ETIQUETAS DE ANOMALÍA MEJORADAS")
        print("="*55)
        
        df = self.df_caracteristicas.copy()
        
        # Seleccionar características numéricas
        numeric_data = df.select_dtypes(include=[np.number])
        
        # Limpiar datos
        numeric_data = numeric_data.fillna(numeric_data.median())
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
        numeric_data = numeric_data.fillna(numeric_data.median())
        
        print(f"   - Usando {len(numeric_data.columns)} características numéricas")
        
        # Normalizar con RobustScaler (más robusto a outliers)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(numeric_data)
        
        # Isolation Forest con parámetros optimizados
        iso_forest = IsolationForest(
            contamination=0.05,
            n_estimators=200,
            max_samples='auto',
            random_state=42,
            bootstrap=True
        )
        anomalias_iso = iso_forest.fit_predict(X_scaled)
        
        # Local Outlier Factor mejorado
        lof = LocalOutlierFactor(
            contamination=0.05,
            n_neighbors=20,
            metric='manhattan',
            novelty=True
        )
        lof.fit(X_scaled)
        anomalias_lof = lof.predict(X_scaled)
        
        # Combinar con lógica mejorada (más conservadora)
        etiquetas_finales = ((anomalias_iso == -1) & (anomalias_lof == -1)).astype(int)
        
        print(f"   - Isolation Forest: {(anomalias_iso == -1).sum()} anomalías")
        print(f"   - LOF: {(anomalias_lof == -1).sum()} anomalías")
        print(f"   - Consenso final: {etiquetas_finales.sum()} anomalías")
        
        # Agregar etiquetas al dataset
        df['anomalia'] = etiquetas_finales
        self.df_caracteristicas = df
        
        return etiquetas_finales
    
    def optimizar_modelos_avanzados(self):
        """
        Optimizar todos los modelos, especialmente CatBoost
        """
        print("\n🤖 OPTIMIZANDO MODELOS AVANZADOS")
        print("="*50)
        
        if not modelos_avanzados_disponibles:
            print("⚠️ Modelos avanzados no disponibles")
            return {}
        
        # Preparar datos
        X_features = self.df_caracteristicas.select_dtypes(include=[np.number]).drop('anomalia', axis=1, errors='ignore')
        y_labels = self.df_caracteristicas['anomalia']
        
        # Limpiar datos
        X_features = X_features.fillna(X_features.median())
        X_features = X_features.replace([np.inf, -np.inf], np.nan)
        X_features = X_features.fillna(X_features.median())
        
        print(f"✓ Datos preparados: {X_features.shape}")
        
        # Verificar balance de clases
        if y_labels.sum() < 10:
            print("⚠️ Muy pocas anomalías para entrenamiento supervisado")
            return {}
        
        # Optimizar CatBoost (mejor modelo actual)
        print("\n🔧 Optimizando CatBoost...")
        catboost_optimizado = self._optimizar_catboost(X_features, y_labels)
        self.modelos_optimizados['catboost'] = catboost_optimizado
        
        # Optimizar LightGBM
        print("\n🔧 Optimizando LightGBM...")
        lightgbm_optimizado = self._optimizar_lightgbm(X_features, y_labels)
        self.modelos_optimizados['lightgbm'] = lightgbm_optimizado
        
        # Optimizar Random Forest
        print("\n🔧 Optimizando Random Forest...")
        rf_optimizado = self._optimizar_random_forest(X_features, y_labels)
        self.modelos_optimizados['random_forest'] = rf_optimizado
        
        return self.modelos_optimizados
    
    def _optimizar_catboost(self, X, y):
        """
        Optimizar CatBoost con parámetros específicos para datos desbalanceados
        """
        # Parámetros optimizados para datos desbalanceados
        catboost_optimizado = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
            class_weights=[1, 10],  # Peso mayor para anomalías
            eval_metric='F1',
            early_stopping_rounds=50
        )
        
        # Validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            catboost_optimizado.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            y_pred = catboost_optimizado.predict(X_val)
            score = f1_score(y_val, y_pred)
            scores.append(score)
        
        print(f"   ✓ CatBoost CV F1-Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        return catboost_optimizado
    
    def _optimizar_lightgbm(self, X, y):
        """
        Optimizar LightGBM
        """
        lightgbm_optimizado = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        
        # Validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            lightgbm_optimizado.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            y_pred = lightgbm_optimizado.predict(X_val)
            score = f1_score(y_val, y_pred)
            scores.append(score)
        
        print(f"   ✓ LightGBM CV F1-Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        return lightgbm_optimizado
    
    def _optimizar_random_forest(self, X, y):
        """
        Optimizar Random Forest
        """
        rf_optimizado = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            rf_optimizado.fit(X_train, y_train)
            y_pred = rf_optimizado.predict(X_val)
            score = f1_score(y_val, y_pred)
            scores.append(score)
        
        print(f"   ✓ Random Forest CV F1-Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        return rf_optimizado
    
    def evaluar_modelos_optimizados(self):
        """
        Evaluar modelos optimizados con validación temporal
        """
        print("\n📊 EVALUANDO MODELOS OPTIMIZADOS")
        print("="*50)
        
        if not self.modelos_optimizados:
            print("⚠️ No hay modelos optimizados para evaluar")
            return {}
        
        X_features = self.df_caracteristicas.select_dtypes(include=[np.number]).drop('anomalia', axis=1, errors='ignore')
        y_labels = self.df_caracteristicas['anomalia']
        
        # Limpiar datos
        X_features = X_features.fillna(X_features.median())
        X_features = X_features.replace([np.inf, -np.inf], np.nan)
        X_features = X_features.fillna(X_features.median())
        
        # División temporal (últimos 20% para test)
        split_idx = int(len(X_features) * 0.8)
        X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
        y_train, y_test = y_labels.iloc[:split_idx], y_labels.iloc[split_idx:]
        
        print(f"✓ División temporal: {len(X_train)} train, {len(X_test)} test")
        
        resultados_evaluacion = {}
        
        for nombre, modelo in self.modelos_optimizados.items():
            print(f"\nEvaluando {nombre}...")
            
            try:
                # Entrenar modelo
                modelo.fit(X_train, y_train)
                
                # Predicciones
                y_pred = modelo.predict(X_test)
                y_pred_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, 'predict_proba') else None
                
                # Métricas
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                
                resultados_evaluacion[nombre] = {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"   ✓ F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                
            except Exception as e:
                print(f"   ⚠️ Error evaluando {nombre}: {e}")
        
        self.resultados_ml = resultados_evaluacion
        return resultados_evaluacion
    
    def optimizar_prophet_solar(self):
        """
        Optimizar Prophet específicamente para datos solares
        """
        print("\n🔮 OPTIMIZANDO PROPHET PARA DATOS SOLARES")
        print("="*55)
        
        if not prophet_disponible:
            print("⚠️ Prophet no disponible")
            return None
        
        # Preparar datos para Prophet
        df_prophet = self.df_original.copy()
        
        # NUEVO: Detección robusta de columna de potencia
        power_col = None
        posibles = [c for c in df_prophet.columns if 'ac_power' in c.lower()]
        if posibles:
            power_col = posibles[0]
        elif 'total_ac_power' in df_prophet.columns:
            power_col = 'total_ac_power'
        elif 'ac_power' in df_prophet.columns:
            power_col = 'ac_power'
        else:
            # Intentar crearla sumando columnas que contengan 'ac_power'
            ac_cols = [c for c in df_prophet.columns if 'ac_power' in c]
            if ac_cols:
                df_prophet['total_ac_power'] = df_prophet[ac_cols].sum(axis=1)
                power_col = 'total_ac_power'
        
        if not power_col:
            print("⚠️ No se encontró columna de potencia AC")
            return None
        
        print(f"Usando columna de potencia: {power_col}")
        
        # Preparar datos
        df_prophet = df_prophet[['measured_on', power_col]].dropna()
        df_prophet = df_prophet[df_prophet[power_col] > 0]
        
        # Filtros mejorados
        Q1 = df_prophet[power_col].quantile(0.25)
        Q3 = df_prophet[power_col].quantile(0.75)
        IQR = Q3 - Q1
        df_prophet = df_prophet[
            (df_prophet[power_col] >= Q1 - 1.5 * IQR) & 
            (df_prophet[power_col] <= Q3 + 1.5 * IQR)
        ]
        
        # Resampleo diario
        df_hourly = df_prophet.set_index('measured_on').resample('D').mean().dropna().reset_index()
        df_prophet_final = df_hourly.rename(columns={'measured_on': 'ds', power_col: 'y'})[['ds', 'y']]
        df_prophet_final = df_prophet_final[df_prophet_final['y'] > 5]
        
        if len(df_prophet_final) < 30:
            print(f"⚠️ Datos insuficientes para Prophet: {len(df_prophet_final)} < 30")
            return None
        
        print(f"✓ Datos finales para Prophet: {len(df_prophet_final)} registros")
        
        try:
            # Prophet optimizado para datos solares
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',  # Mejor para datos solares
                changepoint_prior_scale=0.05,  # Menos cambios
                seasonality_prior_scale=10.0,  # Estacionalidad más fuerte
                holidays_prior_scale=10.0,
                seasonality_fourier_terms=10
            )
            
            # Agregar regresores si están disponibles
            if 'ambient_temperature_o_149575' in self.df_original.columns:
                # Preparar datos de temperatura para Prophet
                temp_data = self.df_original[['measured_on', 'ambient_temperature_o_149575']].copy()
                temp_data = temp_data.set_index('measured_on').resample('D').mean().reset_index()
                temp_data = temp_data.rename(columns={'measured_on': 'ds', 'ambient_temperature_o_149575': 'temperatura'})
                
                # Combinar con datos de Prophet
                df_prophet_final = df_prophet_final.merge(temp_data, on='ds', how='left')
                df_prophet_final['temperatura'] = df_prophet_final['temperatura'].fillna(df_prophet_final['temperatura'].mean())
                
                model.add_regressor('temperatura')
                print("✓ Regresor de temperatura agregado")
            
            # Entrenar modelo
            print("Entrenando modelo Prophet optimizado...")
            model.fit(df_prophet_final)
            
            # Predicciones futuras
            print("Generando predicciones futuras...")
            future = model.make_future_dataframe(periods=360)
            
            # Agregar regresores al futuro si están disponibles
            if 'temperatura' in df_prophet_final.columns:
                # Simular temperatura futura (promedio histórico)
                temp_media = df_prophet_final['temperatura'].mean()
                future['temperatura'] = temp_media
            
            forecast = model.predict(future)
            
            # Evitar valores negativos
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
            
            # Evaluación del modelo
            y_true = df_prophet_final['y'].reset_index(drop=True)
            y_pred = forecast['yhat'][:len(y_true)].reset_index(drop=True)
            
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            # SMAPE mejorado
            def smape(y_true, y_pred):
                return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
            
            smape_val = smape(y_true, y_pred)
            
            print(f"✅ Modelo Prophet optimizado entrenado:")
            print(f"   - MAE: {mae:.2f}")
            print(f"   - RMSE: {rmse:.2f}")
            print(f"   - SMAPE: {smape_val:.2f}%")
            
            self.resultados_prophet = {
                'model': model,
                'forecast': forecast,
                'data_used': df_prophet_final,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'smape': smape_val
                }
            }
            
            return self.resultados_prophet
            
        except Exception as e:
            print(f"⚠️ Error en Prophet optimizado: {e}")
            return None
    
    def calcular_score_sistema_mejorado(self):
        """
        Calcular score del sistema mejorado
        """
        print("\n🎯 CALCULANDO SCORE DEL SISTEMA MEJORADO")
        print("="*55)
        
        score = 0
        
        # Puntuación por características avanzadas
        if self.df_caracteristicas is not None:
            num_caracteristicas = len(self.df_caracteristicas.columns)
            if num_caracteristicas > 50:
                score += 25
                print("✓ Características avanzadas: +25 puntos")
            elif num_caracteristicas > 20:
                score += 15
                print("✓ Características intermedias: +15 puntos")
            else:
                score += 5
                print("✓ Características básicas: +5 puntos")
        
        # Puntuación por modelos optimizados
        if self.modelos_optimizados:
            score += 25
            print("✓ Modelos optimizados: +25 puntos")
        
        # Puntuación por resultados ML
        if self.resultados_ml:
            mejor_f1 = max([res['f1'] for res in self.resultados_ml.values()])
            if mejor_f1 > 0.7:
                score += 25
                print(f"✓ Excelente F1-Score ({mejor_f1:.3f}): +25 puntos")
            elif mejor_f1 > 0.6:
                score += 15
                print(f"✓ Bueno F1-Score ({mejor_f1:.3f}): +15 puntos")
            else:
                score += 5
                print(f"✓ Básico F1-Score ({mejor_f1:.3f}): +5 puntos")
        
        # Puntuación por Prophet
        if self.resultados_prophet:
            smape = self.resultados_prophet['metrics']['smape']
            if smape < 15:
                score += 25
                print(f"✓ Excelente SMAPE ({smape:.1f}%): +25 puntos")
            elif smape < 20:
                score += 15
                print(f"✓ Bueno SMAPE ({smape:.1f}%): +15 puntos")
            else:
                score += 5
                print(f"✓ Básico SMAPE ({smape:.1f}%): +5 puntos")
        
        self.score_sistema = score
        print(f"\n🏆 SCORE FINAL DEL SISTEMA MEJORADO: {score}/100")
        
        if score >= 75:
            print("   Estado: 🟢 EXCELENTE - Sistema completamente optimizado")
        elif score >= 50:
            print("   Estado: 🟡 BUENO - Sistema bien optimizado")
        else:
            print("   Estado: 🔴 BÁSICO - Necesita más optimización")
        
        return score
    
    def generar_reporte_mejorado(self):
        """
        Generar reporte completo del sistema mejorado
        """
        print("\n📋 GENERANDO REPORTE MEJORADO")
        print("="*45)
        
        reporte = {
            'timestamp': pd.Timestamp.now(),
            'score_sistema': self.score_sistema,
            'datos': {
                'registros_totales': len(self.df_caracteristicas) if self.df_caracteristicas is not None else 0,
                'caracteristicas_totales': len(self.df_caracteristicas.columns) if self.df_caracteristicas is not None else 0,
                'anomalias_detectadas': self.df_caracteristicas['anomalia'].sum() if self.df_caracteristicas is not None else 0
            },
            'modelos': {
                'optimizados': len(self.modelos_optimizados),
                'mejor_f1_score': max([res['f1'] for res in self.resultados_ml.values()]) if self.resultados_ml else 0
            },
            'prophet': {
                'smape': self.resultados_prophet['metrics']['smape'] if self.resultados_prophet else None
            }
        }
        
        print("🏆 REPORTE MEJORADO COMPLETADO")
        print("="*40)
        print(f"📅 Generado: {reporte['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Score del sistema: {reporte['score_sistema']}/100")
        print(f"📊 Registros procesados: {reporte['datos']['registros_totales']:,}")
        print(f"🔧 Características: {reporte['datos']['caracteristicas_totales']}")
        print(f"🚨 Anomalías detectadas: {reporte['datos']['anomalias_detectadas']}")
        print(f"🤖 Modelos optimizados: {reporte['modelos']['optimizados']}")
        print(f"📈 Mejor F1-Score: {reporte['modelos']['mejor_f1_score']:.3f}")
        if reporte['prophet']['smape']:
            print(f"🔮 SMAPE Prophet: {reporte['prophet']['smape']:.2f}%")
        
        return reporte
    
    def ejecutar_sistema_completo(self, ruta_datos=None):
        """
        Ejecutar el sistema completo mejorado
        """
        print("🚀 EJECUTANDO SISTEMA MEJORADO V2")
        print("="*50)
        
        # 1. Cargar datos
        self.cargar_y_preparar_datos(ruta_datos)
        
        # 2. Aplicar características avanzadas
        self.aplicar_caracteristicas_avanzadas_corregidas()
        
        # 3. Crear etiquetas de anomalía
        self.crear_etiquetas_anomalia_mejoradas()
        
        # 4. Optimizar modelos
        self.optimizar_modelos_avanzados()
        
        # 5. Evaluar modelos
        self.evaluar_modelos_optimizados()
        
        # 6. Optimizar Prophet
        self.optimizar_prophet_solar()
        
        # 7. Calcular score
        self.calcular_score_sistema_mejorado()
        
        # 8. Generar reporte
        reporte = self.generar_reporte_mejorado()
        
        print("\n✅ SISTEMA MEJORADO V2 COMPLETADO")
        print("="*50)
        
        return {
            'sistema': self,
            'reporte': reporte,
            'datos': self.df_caracteristicas,
            'modelos': self.modelos_optimizados,
            'resultados_ml': self.resultados_ml,
            'prophet': self.resultados_prophet
        }

# Función de conveniencia para ejecutar el sistema
def ejecutar_sistema_mejorado_v2(ruta_datos=None):
    """
    Función de conveniencia para ejecutar el sistema mejorado V2
    """
    sistema = SistemaMejoradoV2()
    return sistema.ejecutar_sistema_completo(ruta_datos)

if __name__ == "__main__":
    # Ejecutar sistema si se llama directamente
    resultados = ejecutar_sistema_mejorado_v2()
    print("\n🎉 Sistema mejorado V2 ejecutado exitosamente!") 
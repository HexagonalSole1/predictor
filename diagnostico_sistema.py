#!/usr/bin/env python3
"""
Script de diagnóstico para el sistema de detección de anomalías
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("🔍 DIAGNÓSTICO DEL SISTEMA")
print("="*40)

# 1. Verificar datos
print("\n1️⃣ VERIFICANDO DATOS...")
try:
    # Cargar datos
    environment_data = pd.read_csv('environment_data.csv')
    irradiance_data = pd.read_csv('irradiance_data.csv')
    electrical_data = pd.read_csv('electrical_data.csv')
    
    print(f"✅ Datos cargados:")
    print(f"   - Environment: {environment_data.shape}")
    print(f"   - Irradiance: {irradiance_data.shape}")
    print(f"   - Electrical: {electrical_data.shape}")
    
    # Combinar datos
    df_inv1 = electrical_data[['measured_on', 'inv_01_dc_current_inv_149579', 'inv_01_dc_voltage_inv_149580', 
                              'inv_01_ac_current_inv_149581', 'inv_01_ac_voltage_inv_149582', 'inv_01_ac_power_inv_149583']].copy()
    df_inv1.columns = ['measured_on', 'dc_current', 'dc_voltage', 'ac_current', 'ac_voltage', 'ac_power']
    
    environment_data['measured_on'] = pd.to_datetime(environment_data['measured_on'])
    irradiance_data['measured_on'] = pd.to_datetime(irradiance_data['measured_on'])
    df_inv1['measured_on'] = pd.to_datetime(df_inv1['measured_on'])
    
    df_combined = environment_data.merge(irradiance_data, on='measured_on', how='inner')
    df_final = df_combined.merge(df_inv1, on='measured_on', how='inner')
    
    print(f"✅ Datos combinados: {df_final.shape}")
    
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    exit(1)

# 2. Verificar características
print("\n2️⃣ VERIFICANDO CARACTERÍSTICAS...")
try:
    df_for_features = df_final.copy()
    df_for_features = df_for_features.set_index('measured_on')
    
    # Características básicas
    df_for_features['hour'] = df_for_features.index.hour
    df_for_features['day_of_week'] = df_for_features.index.dayofweek
    df_for_features['month'] = df_for_features.index.month
    df_for_features['total_ac_power'] = df_for_features['ac_power']
    
    if 'poa_irradiance_o_149574' in df_for_features.columns:
        df_for_features['eficiencia_basica'] = df_for_features['total_ac_power'] / (df_for_features['poa_irradiance_o_149574'] + 1e-6)
    
    print(f"✅ Características básicas creadas: {df_for_features.shape}")
    
except Exception as e:
    print(f"❌ Error creando características: {e}")
    exit(1)

# 3. Verificar etiquetas de anomalía
print("\n3️⃣ VERIFICANDO ETIQUETAS DE ANOMALÍA...")
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    
    # Seleccionar características numéricas
    numeric_data = df_for_features.select_dtypes(include=[np.number])
    print(f"   - Características numéricas: {numeric_data.shape}")
    
    # Limpiar datos
    numeric_data = numeric_data.fillna(numeric_data.median())
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.fillna(numeric_data.median())
    print(f"   - Datos limpios: {numeric_data.shape}")
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)
    print(f"   - Datos escalados: {X_scaled.shape}")
    
    # Isolation Forest
    print("   - Aplicando Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalias_iso = iso_forest.fit_predict(X_scaled)
    print(f"   - Isolation Forest completado: {(anomalias_iso == -1).sum()} anomalías")
    
    # LOF
    print("   - Aplicando Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination=0.05, novelty=True)
    lof.fit(X_scaled)
    anomalias_lof = lof.predict(X_scaled)
    print(f"   - LOF completado: {(anomalias_lof == -1).sum()} anomalías")
    
    # Combinar
    etiquetas_finales = ((anomalias_iso == -1) | (anomalias_lof == -1)).astype(int)
    print(f"   - Consenso final: {etiquetas_finales.sum()} anomalías")
    
    print("✅ Etiquetas de anomalía creadas exitosamente")
    
except Exception as e:
    print(f"❌ Error creando etiquetas: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. Verificar módulos personalizados
print("\n4️⃣ VERIFICANDO MÓDULOS PERSONALIZADOS...")
try:
    from mejoras_ingenieria_caracteristicas import MejorasIngenieriaCaracteristicas
    from mejoras_modelo_anomalias import MejorasModeloAnomalias
    
    mejorador_caracteristicas = MejorasIngenieriaCaracteristicas()
    mejorador_modelo = MejorasModeloAnomalias()
    
    print("✅ Módulos personalizados funcionando")
    
except Exception as e:
    print(f"❌ Error con módulos personalizados: {e}")

# 5. Verificar Prophet
print("\n5️⃣ VERIFICANDO PROPHET...")
try:
    from prophet import Prophet
    print("✅ Prophet disponible")
except Exception as e:
    print(f"❌ Error con Prophet: {e}")

print("\n✅ DIAGNÓSTICO COMPLETADO")
print("El sistema debería funcionar correctamente") 
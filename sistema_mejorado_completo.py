"""
SISTEMA MEJORADO COMPLETO DE DETECCIÓN DE ANOMALÍAS
===================================================

Este script implementa todas las mejoras propuestas para el sistema de detección
de anomalías en paneles fotovoltaicos, combinando:

1. Ingeniería de características avanzada
2. Modelos de machine learning mejorados
3. Sistema de alertas inteligente
4. Validación y evaluación robusta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar las clases de mejoras
from mejoras_ingenieria_caracteristicas import MejorasIngenieriaCaracteristicas
from mejoras_modelo_anomalias import MejorasModeloAnomalias

class SistemaMejoradoCompleto:
    """
    Sistema completo mejorado para detección de anomalías
    """
    
    def __init__(self):
        self.mejorador_caracteristicas = MejorasIngenieriaCaracteristicas()
        self.mejorador_modelo = MejorasModeloAnomalias()
        self.datos_procesados = None
        self.modelos_entrenados = None
        self.resultados = None
    
    def cargar_y_preparar_datos(self, ruta_datos):
        """
        Cargar y preparar los datos iniciales
        """
        print("📊 CARGANDO Y PREPARANDO DATOS")
        print("="*50)
        
        # Cargar datos (ajustar según tu estructura)
        try:
            df = pd.read_csv(ruta_datos)
            
            # Verificar si existe hour_rounded o measured_on
            if 'hour_rounded' in df.columns:
                df['measured_on'] = pd.to_datetime(df['hour_rounded'])
                df = df.drop('hour_rounded', axis=1)
            elif 'measured_on' in df.columns:
                df['measured_on'] = pd.to_datetime(df['measured_on'])
            else:
                print("❌ No se encontró columna de fecha (hour_rounded o measured_on)")
                return None
            
            df = df.set_index('measured_on')
            
            print(f"✓ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
            print(f"✓ Período: {df.index.min()} a {df.index.max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def aplicar_ingenieria_caracteristicas_completa(self, df):
        """
        Aplicar ingeniería de características completa
        """
        print("\n🔧 APLICANDO INGENIERÍA DE CARACTERÍSTICAS COMPLETA")
        print("="*65)
        
        # Aplicar todas las mejoras de características
        df_mejorado = self.mejorador_caracteristicas.aplicar_todas_mejoras(df)
        
        # Crear etiquetas de anomalía antes de seleccionar características
        etiquetas = self.crear_etiquetas_anomalia(df_mejorado)
        df_mejorado['anomalia'] = etiquetas
        
        # Seleccionar características más importantes
        caracteristicas_importantes = self.mejorador_caracteristicas.seleccionar_caracteristicas_importantes(
            df_mejorado, 
            target='anomalia', 
            metodo='correlacion', 
            top_n=50
        )
        
        # Filtrar solo características importantes
        df_final = df_mejorado[caracteristicas_importantes]
        
        print(f"✓ Características finales seleccionadas: {len(df_final.columns)}")
        
        self.datos_procesados = df_final
        return df_final
    
    def crear_etiquetas_anomalia(self, df, metodo='consenso'):
        """
        Crear etiquetas de anomalía usando múltiples métodos
        """
        print(f"\n🏷️ CREANDO ETIQUETAS DE ANOMALÍA ({metodo})")
        print("="*55)
        
        # Verificar si ya existen etiquetas de anomalía en los datos
        columnas_anomalia = [col for col in df.columns if 'anomaly' in col.lower()]
        
        if columnas_anomalia:
            print(f"✓ Usando etiquetas existentes: {columnas_anomalia}")
            
            # Combinar etiquetas existentes
            etiquetas_combinadas = df[columnas_anomalia].astype(int).sum(axis=1)
            
            if metodo == 'consenso':
                # Anomalía si al menos 2 métodos la detectan
                etiquetas_finales = (etiquetas_combinadas >= 2).astype(int)
            elif metodo == 'mayoria':
                # Anomalía si al menos 1 método la detecta
                etiquetas_finales = (etiquetas_combinadas >= 1).astype(int)
            else:
                # Usar solo la primera columna de anomalía
                etiquetas_finales = df[columnas_anomalia[0]].astype(int)
        else:
            print("⚠️ No se encontraron etiquetas de anomalía, creando nuevas...")
            
            # Métodos de detección básica para crear etiquetas
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            
            anomalias_iso = IsolationForest(contamination=0.01, random_state=42)
            anomalias_lof = LocalOutlierFactor(contamination=0.01, novelty=True)
            
            # Detectar anomalías
            pred_iso = anomalias_iso.fit_predict(df.select_dtypes(include=[np.number]))
            pred_lof = anomalias_lof.fit(df.select_dtypes(include=[np.number])).predict(df.select_dtypes(include=[np.number]))
            
            # Convertir a etiquetas binarias
            etiquetas_iso = (pred_iso == -1).astype(int)
            etiquetas_lof = (pred_lof == -1).astype(int)
            
            if metodo == 'consenso':
                # Anomalía si ambos métodos la detectan
                etiquetas_finales = ((etiquetas_iso + etiquetas_lof) >= 1).astype(int)
            elif metodo == 'mayoria':
                # Anomalía si al menos uno la detecta
                etiquetas_finales = ((etiquetas_iso + etiquetas_lof) >= 1).astype(int)
            else:
                etiquetas_finales = etiquetas_iso
        
        print(f"✓ Anomalías detectadas: {etiquetas_finales.sum()} ({etiquetas_finales.sum()/len(etiquetas_finales)*100:.2f}%)")
        
        return etiquetas_finales
    
    def entrenar_modelos_mejorados(self, X, y):
        """
        Entrenar modelos mejorados
        """
        print("\n🤖 ENTRENANDO MODELOS MEJORADOS")
        print("="*45)
        
        # Aplicar mejoras del modelo
        resultados_modelo = self.mejorador_modelo.aplicar_todas_mejoras(X, y, metodo='supervisado')
        
        # Entrenar modelos individuales
        modelos_entrenados = {}
        for nombre, modelo in resultados_modelo['modelos'].items():
            print(f"   Entrenando {nombre}...")
            modelo.fit(X, y)
            modelos_entrenados[nombre] = modelo
        
        # Crear ensemble de votación
        voting_clf = self.mejorador_modelo.crear_sistema_voting(X, y)
        voting_clf.fit(X, y)
        modelos_entrenados['ensemble_voting'] = voting_clf
        
        self.modelos_entrenados = modelos_entrenados
        return modelos_entrenados
    
    def evaluar_sistema_completo(self, X_test, y_test):
        """
        Evaluar el sistema completo
        """
        print("\n📊 EVALUANDO SISTEMA COMPLETO")
        print("="*45)
        
        resultados_evaluacion = {}
        
        for nombre, modelo in self.modelos_entrenados.items():
            print(f"\n   Evaluando {nombre}...")
            resultados = self.mejorador_modelo.evaluar_modelo_avanzado(
                modelo, X_test, y_test, nombre
            )
            resultados_evaluacion[nombre] = resultados
        
        # Comparar modelos
        self.comparar_modelos(resultados_evaluacion)
        
        self.resultados = resultados_evaluacion
        return resultados_evaluacion
    
    def comparar_modelos(self, resultados_evaluacion):
        """
        Comparar rendimiento de todos los modelos
        """
        print("\n📈 COMPARACIÓN DE MODELOS")
        print("="*35)
        
        # Crear DataFrame de comparación
        comparacion = []
        for nombre, resultados in resultados_evaluacion.items():
            comparacion.append({
                'Modelo': nombre,
                'Accuracy': resultados['accuracy'],
                'Precision': resultados['precision'],
                'Recall': resultados['recall'],
                'F1-Score': resultados['f1']
            })
        
        df_comparacion = pd.DataFrame(comparacion)
        df_comparacion = df_comparacion.sort_values('F1-Score', ascending=False)
        
        print(df_comparacion.to_string(index=False))
        
        # Graficar comparación
        self.graficar_comparacion_modelos(df_comparacion)
        
        return df_comparacion
    
    def graficar_comparacion_modelos(self, df_comparacion):
        """
        Graficar comparación de modelos
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].bar(df_comparacion['Modelo'], df_comparacion['Accuracy'])
        axes[0,0].set_title('Accuracy por Modelo')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Precision
        axes[0,1].bar(df_comparacion['Modelo'], df_comparacion['Precision'])
        axes[0,1].set_title('Precision por Modelo')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[1,0].bar(df_comparacion['Modelo'], df_comparacion['Recall'])
        axes[1,0].set_title('Recall por Modelo')
        axes[1,0].set_ylabel('Recall')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1,1].bar(df_comparacion['Modelo'], df_comparacion['F1-Score'])
        axes[1,1].set_title('F1-Score por Modelo')
        axes[1,1].set_ylabel('F1-Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def crear_sistema_alertas_final(self):
        """
        Crear sistema de alertas final
        """
        print("\n🚨 CREANDO SISTEMA DE ALERTAS FINAL")
        print("="*45)
        
        sistema_alertas = {
            'configuracion': {
                'umbrales': {
                    'bajo': 0.3,
                    'medio': 0.6,
                    'alto': 0.8,
                    'critico': 0.95
                },
                'acciones': {
                    'bajo': 'Monitoreo continuo',
                    'medio': 'Investigación técnica',
                    'alto': 'Intervención inmediata',
                    'critico': 'Emergencia - Parada del sistema'
                }
            },
            'filtros': {
                'ventana_tiempo': 24,  # horas
                'min_alertas_consecutivas': 2,
                'periodo_cooldown': 6  # horas
            },
            'notificaciones': {
                'email': True,
                'sms': True,
                'dashboard': True,
                'log_sistema': True
            }
        }
        
        print("✓ Sistema de alertas configurado")
        return sistema_alertas
    
    def generar_reporte_final(self):
        """
        Generar reporte final del sistema
        """
        print("\n📋 GENERANDO REPORTE FINAL")
        print("="*40)
        
        reporte = {
            'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'datos_procesados': {
                'registros_originales': len(self.datos_procesados) if self.datos_procesados is not None else 0,
                'caracteristicas_finales': len(self.datos_procesados.columns) if self.datos_procesados is not None else 0,
                'anomalias_detectadas': 0  # Se calcularía con los datos reales
            },
            'modelos_entrenados': len(self.modelos_entrenados) if self.modelos_entrenados is not None else 0,
            'mejor_modelo': None,
            'metricas_mejores': None
        }
        
        if self.resultados is not None:
            # Encontrar mejor modelo por F1-Score
            mejor_f1 = 0
            mejor_modelo = None
            
            for nombre, resultados in self.resultados.items():
                if resultados['f1'] > mejor_f1:
                    mejor_f1 = resultados['f1']
                    mejor_modelo = nombre
            
            reporte['mejor_modelo'] = mejor_modelo
            reporte['metricas_mejores'] = self.resultados[mejor_modelo]
        
        print("✓ Reporte final generado")
        return reporte
    
    def ejecutar_sistema_completo(self, ruta_datos):
        """
        Ejecutar el sistema completo mejorado
        """
        print("🚀 EJECUTANDO SISTEMA COMPLETO MEJORADO")
        print("="*60)
        
        # 1. Cargar datos
        df = self.cargar_y_preparar_datos(ruta_datos)
        if df is None:
            return None
        
        # 2. Aplicar ingeniería de características
        df_mejorado = self.aplicar_ingenieria_caracteristicas_completa(df)
        
        # 3. Crear etiquetas de anomalía
        etiquetas = self.crear_etiquetas_anomalia(df_mejorado)
        df_mejorado['anomalia'] = etiquetas
        
        # 4. Dividir datos (ajustar según necesidad)
        from sklearn.model_selection import train_test_split
        
        # Seleccionar solo columnas numéricas para el modelo
        columnas_numericas = df_mejorado.select_dtypes(include=[np.number]).columns
        X = df_mejorado[columnas_numericas].drop('anomalia', axis=1, errors='ignore')
        y = df_mejorado['anomalia']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 5. Entrenar modelos
        modelos = self.entrenar_modelos_mejorados(X_train, y_train)
        
        # 6. Evaluar sistema
        resultados = self.evaluar_sistema_completo(X_test, y_test)
        
        # 7. Crear sistema de alertas
        sistema_alertas = self.crear_sistema_alertas_final()
        
        # 8. Generar reporte
        reporte = self.generar_reporte_final()
        
        print("\n✅ SISTEMA COMPLETO EJECUTADO EXITOSAMENTE")
        print("="*60)
        
        return {
            'datos_procesados': df_mejorado,
            'modelos_entrenados': modelos,
            'resultados_evaluacion': resultados,
            'sistema_alertas': sistema_alertas,
            'reporte_final': reporte
        }

# Función principal para ejecutar el sistema
def main():
    """
    Función principal para ejecutar el sistema mejorado
    """
    print("🔋 SISTEMA MEJORADO DE DETECCIÓN DE ANOMALÍAS EN PANELES SOLARES")
    print("="*75)
    
    # Crear instancia del sistema
    sistema = SistemaMejoradoCompleto()
    
    # Ejecutar sistema completo
    # Ajustar la ruta según tus datos
    ruta_datos = "anomalias_detectadas_planta_solar.csv"  # Ajustar según tu archivo
    
    try:
        resultados = sistema.ejecutar_sistema_completo(ruta_datos)
        
        if resultados is not None:
            print("\n🎉 SISTEMA EJECUTADO CON ÉXITO")
            print("="*40)
            print(f"📊 Datos procesados: {len(resultados['datos_procesados'])} registros")
            print(f"🤖 Modelos entrenados: {len(resultados['modelos_entrenados'])}")
            print(f"📈 Mejor modelo: {resultados['reporte_final']['mejor_modelo']}")
            print(f"🚨 Sistema de alertas: Configurado")
            
        else:
            print("❌ Error en la ejecución del sistema")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de tener los archivos de datos correctos")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class MejorasModeloAnomalias:
    """
    Clase para implementar mejoras en el modelo de detección de anomalías
    """
    
    def __init__(self):
        self.modelos_entrenados = {}
        self.escaladores = {}
        self.mejores_parametros = {}
    
    def crear_ensemble_avanzado(self, X, y=None, metodo='supervisado'):
        """
        Crear ensemble avanzado de modelos de detección de anomalías
        """
        print("🤖 CREANDO ENSEMBLE AVANZADO DE MODELOS")
        print("="*55)
        
        modelos = {}
        
        if metodo == 'no_supervisado':
            # ========================================
            # 1. ISOLATION FOREST MEJORADO
            # ========================================
            iso_forest = IsolationForest(
                contamination=0.01,
                n_estimators=200,
                max_samples='auto',
                random_state=42,
                bootstrap=True
            )
            modelos['isolation_forest'] = iso_forest
            
            # ========================================
            # 2. LOCAL OUTLIER FACTOR
            # ========================================
            lof = LocalOutlierFactor(
                contamination=0.01,
                n_neighbors=20,
                metric='manhattan',
                novelty=True
            )
            modelos['local_outlier_factor'] = lof
            
            # ========================================
            # 3. ONE-CLASS SVM
            # ========================================
            oc_svm = OneClassSVM(
                kernel='rbf',
                nu=0.01,
                gamma='scale'
            )
            modelos['one_class_svm'] = oc_svm
            
            # ========================================
            # 4. DBSCAN PARA CLUSTERING
            # ========================================
            dbscan = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            )
            modelos['dbscan'] = dbscan
        
        elif metodo == 'supervisado':
            # ========================================
            # 1. RANDOM FOREST
            # ========================================
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            modelos['random_forest'] = rf
            
            # ========================================
            # 2. XGBOOST
            # ========================================
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=99  # Para datos desbalanceados
            )
            modelos['xgboost'] = xgb_model
            
            # ========================================
            # 3. LIGHTGBM
            # ========================================
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight='balanced'
            )
            modelos['lightgbm'] = lgb_model
            
            # ========================================
            # 4. CATBOOST
            # ========================================
            cat_model = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
            modelos['catboost'] = cat_model
        
        self.modelos_entrenados = modelos
        return modelos
    
    def optimizar_hiperparametros(self, X, y, modelo_nombre, cv_folds=5):
        """
        Optimizar hiperparámetros usando validación cruzada temporal
        """
        print(f"\n🔧 OPTIMIZANDO HIPERPARÁMETROS PARA {modelo_nombre.upper()}")
        print("="*60)
        
        # Definir parámetros según el modelo
        if modelo_nombre == 'isolation_forest':
            param_grid = {
                'contamination': [0.005, 0.01, 0.02, 0.05],
                'n_estimators': [100, 200, 300],
                'max_samples': ['auto', 100, 200]
            }
            modelo = IsolationForest(random_state=42)
            
        elif modelo_nombre == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            modelo = RandomForestClassifier(random_state=42, class_weight='balanced')
            
        elif modelo_nombre == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9]
            }
            modelo = xgb.XGBClassifier(random_state=42)
        
        # Validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=modelo,
            param_grid=param_grid,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"✓ Mejores parámetros: {grid_search.best_params_}")
        print(f"✓ Mejor score: {grid_search.best_score_:.4f}")
        
        self.mejores_parametros[modelo_nombre] = grid_search.best_params_
        return grid_search.best_estimator_
    
    def crear_escaladores_robustos(self, X):
        """
        Crear escaladores robustos para diferentes tipos de datos
        """
        print("\n📏 CREANDO ESCALADORES ROBUSTOS")
        print("="*40)
        
        escaladores = {}
        
        # ========================================
        # 1. STANDARD SCALER (para datos normales)
        # ========================================
        scaler_std = StandardScaler()
        escaladores['standard'] = scaler_std
        
        # ========================================
        # 2. ROBUST SCALER (para datos con outliers)
        # ========================================
        scaler_robust = RobustScaler()
        escaladores['robust'] = scaler_robust
        
        # ========================================
        # 3. MIN-MAX SCALER (para datos acotados)
        # ========================================
        from sklearn.preprocessing import MinMaxScaler
        scaler_minmax = MinMaxScaler()
        escaladores['minmax'] = scaler_minmax
        
        # ========================================
        # 4. POWER TRANSFORMER (para datos sesgados)
        # ========================================
        from sklearn.preprocessing import PowerTransformer
        scaler_power = PowerTransformer(method='yeo-johnson')
        escaladores['power'] = scaler_power
        
        self.escaladores = escaladores
        return escaladores
    
    def aplicar_escalado_inteligente(self, X, metodo='robust'):
        """
        Aplicar escalado inteligente según las características de los datos
        """
        print(f"\n🧠 APLICANDO ESCALADO INTELIGENTE ({metodo})")
        print("="*50)
        
        if metodo == 'auto':
            # Detectar automáticamente el mejor escalador
            from scipy import stats
            
            # Calcular asimetría de cada columna
            skewness = stats.skew(X, axis=0)
            outliers_ratio = np.sum(np.abs(stats.zscore(X)) > 3, axis=0) / len(X)
            
            # Decidir escalador basado en características
            if np.mean(skewness) > 1 or np.mean(outliers_ratio) > 0.1:
                metodo = 'robust'
            elif np.mean(skewness) > 0.5:
                metodo = 'power'
            else:
                metodo = 'standard'
        
        escalador = self.escaladores[metodo]
        X_escalado = escalador.fit_transform(X)
        
        print(f"✓ Escalado aplicado: {metodo}")
        print(f"✓ Forma de datos: {X_escalado.shape}")
        
        return X_escalado, escalador
    
    def crear_sistema_voting(self, X, y, pesos=None):
        """
        Crear sistema de votación ponderada
        """
        print("\n🗳️ CREANDO SISTEMA DE VOTACIÓN PONDERADA")
        print("="*50)
        
        from sklearn.ensemble import VotingClassifier
        
        # Si no se especifican pesos, usar pesos basados en rendimiento
        if pesos is None:
            pesos = {'random_forest': 0.3, 'xgboost': 0.3, 'lightgbm': 0.2, 'catboost': 0.2}
        
        # Crear ensemble de clasificadores
        estimators = []
        for nombre, modelo in self.modelos_entrenados.items():
            if nombre in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
                estimators.append((nombre, modelo))
        
        # Sistema de votación
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Probabilidades ponderadas
            weights=[pesos[nombre] for nombre, _ in estimators]
        )
        
        return voting_clf
    
    def crear_deteccion_multinivel(self, X, y=None):
        """
        Crear sistema de detección multinivel
        """
        print("\n🏗️ CREANDO SISTEMA DE DETECCIÓN MULTINIVEL")
        print("="*55)
        
        sistema_multinivel = {}
        
        # ========================================
        # NIVEL 1: DETECCIÓN RÁPIDA (Isolation Forest)
        # ========================================
        print("   Nivel 1: Detección rápida con Isolation Forest")
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        sistema_multinivel['nivel_1'] = iso_forest
        
        # ========================================
        # NIVEL 2: DETECCIÓN PRECISA (LOF + SVM)
        # ========================================
        print("   Nivel 2: Detección precisa con LOF + SVM")
        lof = LocalOutlierFactor(contamination=0.01, novelty=True)
        oc_svm = OneClassSVM(nu=0.01)
        sistema_multinivel['nivel_2'] = {'lof': lof, 'svm': oc_svm}
        
        # ========================================
        # NIVEL 3: CLASIFICACIÓN FINAL (Ensemble)
        # ========================================
        print("   Nivel 3: Clasificación final con ensemble")
        if y is not None:
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
            xgb_model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=99)
            sistema_multinivel['nivel_3'] = {'rf': rf, 'xgb': xgb_model}
        
        return sistema_multinivel
    
    def evaluar_modelo_avanzado(self, modelo, X_test, y_test, nombre_modelo="Modelo"):
        """
        Evaluación avanzada del modelo
        """
        print(f"\n📊 EVALUACIÓN AVANZADA: {nombre_modelo}")
        print("="*50)
        
        # Predicciones
        if hasattr(modelo, 'predict_proba'):
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = modelo.predict(X_test)
            y_pred_proba = None
        
        # Métricas básicas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"✓ Accuracy: {accuracy:.4f}")
        print(f"✓ Precision: {precision:.4f}")
        print(f"✓ Recall: {recall:.4f}")
        print(f"✓ F1-Score: {f1:.4f}")
        
        # ROC-AUC si hay probabilidades
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                print(f"✓ ROC-AUC: {roc_auc:.4f}")
            except:
                print("⚠️ No se pudo calcular ROC-AUC")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📋 Matriz de Confusión:")
        print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Reporte detallado
        print(f"\n📄 Reporte de Clasificación:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def crear_sistema_alertas_inteligente(self, umbrales_dinamicos=True):
        """
        Crear sistema de alertas inteligente
        """
        print("\n🚨 CREANDO SISTEMA DE ALERTAS INTELIGENTE")
        print("="*55)
        
        sistema_alertas = {
            'niveles': {
                'bajo': {'umbral': 0.3, 'accion': 'Monitoreo'},
                'medio': {'umbral': 0.6, 'accion': 'Investigación'},
                'alto': {'umbral': 0.8, 'accion': 'Intervención'},
                'critico': {'umbral': 0.95, 'accion': 'Emergencia'}
            },
            'umbrales_dinamicos': umbrales_dinamicos,
            'filtros_temporales': {
                'ventana_analisis': 24,  # horas
                'min_alertas_consecutivas': 2,
                'cooldown_period': 6  # horas
            }
        }
        
        return sistema_alertas
    
    def aplicar_todas_mejoras(self, X, y=None, metodo='supervisado'):
        """
        Aplicar todas las mejoras del modelo
        """
        print("🚀 APLICANDO TODAS LAS MEJORAS DEL MODELO")
        print("="*55)
        
        # 1. Crear escaladores
        self.crear_escaladores_robustos(X)
        
        # 2. Aplicar escalado inteligente
        X_escalado, escalador = self.aplicar_escalado_inteligente(X, metodo='auto')
        
        # 3. Crear ensemble de modelos
        modelos = self.crear_ensemble_avanzado(X_escalado, y, metodo)
        
        # 4. Optimizar hiperparámetros (si hay datos de entrenamiento)
        if y is not None:
            for nombre_modelo in ['random_forest', 'xgboost']:
                if nombre_modelo in modelos:
                    self.optimizar_hiperparametros(X_escalado, y, nombre_modelo)
        
        # 5. Crear sistema multinivel
        sistema_multinivel = self.crear_deteccion_multinivel(X_escalado, y)
        
        # 6. Crear sistema de alertas
        sistema_alertas = self.crear_sistema_alertas_inteligente()
        
        print(f"\n✅ MEJORAS DEL MODELO COMPLETADAS")
        print(f"   - Modelos creados: {len(modelos)}")
        print(f"   - Escaladores: {len(self.escaladores)}")
        print(f"   - Sistema multinivel: {len(sistema_multinivel)} niveles")
        print(f"   - Sistema de alertas: {len(sistema_alertas['niveles'])} niveles")
        
        return {
            'modelos': modelos,
            'escalador': escalador,
            'sistema_multinivel': sistema_multinivel,
            'sistema_alertas': sistema_alertas,
            'X_escalado': X_escalado
        }

# Ejemplo de uso
if __name__ == "__main__":
    print("🤖 EJEMPLO DE USO DE MEJORAS DEL MODELO")
    print("="*50)
    
    # Aquí cargarías tus datos reales
    # X, y = cargar_tus_datos()
    
    # Aplicar mejoras
    # mejorador = MejorasModeloAnomalias()
    # resultados = mejorador.aplicar_todas_mejoras(X, y)
    
    print("✅ Script de mejoras del modelo listo para usar") 
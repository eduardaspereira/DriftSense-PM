"""
DriftSense-PM: Streamlit Dashboard - Real-time Anomaly Detection & Visualization
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Dashboard académico para monitorização e deteção de anomalias em tempo real.
"""

import os
import yaml
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, kurtosis, pearsonr, ks_2samp
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import subprocess
import time

warnings.filterwarnings('ignore')

# ============================================================================
# 0. INICIALIZAÇÃO SEGURA DO SESSION STATE
# ============================================================================

if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train_pred' not in st.session_state:
    st.session_state.y_train_pred = None
if 'y_test_pred' not in st.session_state:
    st.session_state.y_test_pred = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'df_data' not in st.session_state:
    st.session_state.df_data = None
if 'auto_train_requested' not in st.session_state:
    st.session_state.auto_train_requested = False

# ============================================================================
# 1. SETUP E CONFIGURAÇÃO
# ============================================================================

st.set_page_config(
    page_title="DriftSense-PM Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("DriftSense-PM: Real-Time Anomaly Detection Dashboard")
st.markdown("---")

# Obter diretório do script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Carregar configuração
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs/config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
else:
    st.error("[ERRO] Configuração não encontrada no caminho especificado.")
    st.stop()

# Usar caminhos absolutos
RAW_DIR = os.path.join(SCRIPT_DIR, config['paths']['raw_data_dir'].lstrip('../'))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, config['paths']['processed_dir'].lstrip('../'))
MODELS_DIR = os.path.join(SCRIPT_DIR, config['paths']['models_dir'].lstrip('../'))
RESULTS_DIR = os.path.join(SCRIPT_DIR, config['paths']['results_dir'].lstrip('../'))
WINDOW_SIZE = config['feature_engineering']['window_size']
STEP_SIZE = config['feature_engineering']['step_size']
TAXA_AMOSTRAGEM = config['system']['sampling_rate_hz']

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# 2. FUNÇÕES AUXILIARES - FEATURE ENGINEERING
# ============================================================================

def calcular_frequencia_pico(dados, fs):
    """Calcular frequência de pico do sinal via FFT."""
    n = len(dados)
    if n == 0 or np.all(dados == dados[0]):
        return 0.0
    yf = np.abs(rfft(dados))
    xf = rfftfreq(n, 1 / fs)
    idx_pico = np.argmax(yf[1:]) + 1 if len(yf) > 1 else 0
    return round(xf[idx_pico], 3)

def extrair_features(df_raw, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Extrai features estatísticas e de frequência de janelas de dados observados."""
    colunas_vibracao = ['AccX', 'AccY', 'AccZ']
    
    for col in ['Temp', 'Hum'] + colunas_vibracao:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    result = pd.DataFrame()
    
    if 'Temp' in df_raw.columns:
        result['Temp_Mean'] = df_raw['Temp'].rolling(window=window_size, min_periods=1).mean().round(2)
    if 'Hum' in df_raw.columns:
        result['Hum_Mean'] = df_raw['Hum'].rolling(window=window_size, min_periods=1).mean().round(2)
    
    if 'Scenario' in df_raw.columns:
        result['Scenario'] = df_raw['Scenario'].rolling(window=window_size, min_periods=1).apply(
            lambda x: x.iloc[0] if len(x) > 0 else 'Unknown', raw=False
        )
    
    for eixo in colunas_vibracao:
        if eixo in df_raw.columns:
            rolling_obj = df_raw[eixo].rolling(window=window_size, min_periods=1)
            
            result[f'{eixo}_Mean'] = rolling_obj.mean().round(2)
            result[f'{eixo}_Std'] = rolling_obj.std().round(2)
            result[f'{eixo}_Max'] = rolling_obj.max().round(2)
            result[f'{eixo}_Min'] = rolling_obj.min().round(2)
            result[f'{eixo}_RMS'] = np.sqrt((df_raw[eixo]**2).rolling(window=window_size, min_periods=1).mean()).round(2)
            
            result[f'{eixo}_Skew'] = df_raw[eixo].rolling(window=window_size, min_periods=1).apply(
                lambda x: skew(x) if len(x) > 2 else 0.0, raw=False
            ).round(3)
            result[f'{eixo}_Kurt'] = df_raw[eixo].rolling(window=window_size, min_periods=1).apply(
                lambda x: kurtosis(x) if len(x) > 3 else 0.0, raw=False
            ).round(3)
            
            result[f'{eixo}_PeakFreq_Hz'] = df_raw[eixo].rolling(window=window_size, min_periods=1).apply(
                lambda x: calcular_frequencia_pico(x.values, TAXA_AMOSTRAGEM), raw=False
            )
    
    if step_size > 1:
        result = result.iloc[::step_size].reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)
    
    return result

# ============================================================================
# 3. SIDEBAR - SELEÇÃO DE FONTE DE DADOS E MODELO
# ============================================================================

st.sidebar.markdown("## Configuração de Dados")
data_source = st.sidebar.radio(
    "Fonte de dados:",
    ["Amostras CSV Processadas", "Processar Conjunto de Dados Raw"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Seleção do Modelo")
model_choice = st.sidebar.selectbox(
    "Algoritmo de Deteção:",
    ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"],
    help="LOF: Local Outlier Factor | IF: Isolation Forest | SVM: Support Vector Machine"
)

model_map = {
    "Isolation Forest": "IsolationForest",
    "One-Class SVM": "OneClassSVM",
    "Local Outlier Factor": "LOF"
}
selected_model = model_map[model_choice]

st.sidebar.markdown("---")

st.sidebar.markdown("## Execução da Pipeline")

col_exp1, col_exp2 = st.sidebar.columns(2)

with col_exp1:
    if st.button("Executar Pipeline Completa", key="run_exp", width='stretch'):
        try:
            run_experiment_path = os.path.join(SCRIPT_DIR, "scripts/run_experiment.py")

            with st.spinner("A executar aquisição e treino completo..."):
                result_experiment = subprocess.run(
                    [sys.executable, run_experiment_path],
                    cwd=SCRIPT_DIR,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result_experiment.returncode != 0:
                    st.error(f"[ERRO] Falha na aquisição: {result_experiment.stderr[:500]}")
                else:
                    st.session_state.auto_train_requested = True
                    st.session_state.model = None
                    st.session_state.scaler = None
                    st.session_state.X_train = None
                    st.session_state.X_test = None
                    st.session_state.y_train_pred = None
                    st.session_state.y_test_pred = None
                    st.session_state.feature_cols = None
                    st.success("[OK] Pipeline completa executada com sucesso.")
                    st.info("[INFO] Os resultados analíticos serão gerados automaticamente com os parâmetros escolhidos.")
        except subprocess.TimeoutExpired:
            st.warning("[AVISO] Tempo limite de execução expirado.")
        except Exception as e:
            st.error(f"[ERRO] Instabilidade no subset: {e}")

with col_exp2:
    if st.button("Atualizar Cache", key="refresh_data", width='stretch'):
        st.cache_data.clear()
        st.success("[OK] Cache limpa.")

st.sidebar.markdown("---")

train_test_split_ratio = st.sidebar.slider(
    "Proporção de Divisão (Treino/Teste)",
    min_value=0.5,
    max_value=0.95,
    value=0.8,
    step=0.05
)


st.sidebar.markdown("## Parâmetros de Treino")
st.sidebar.caption("Ajuste os hiperparâmetros de cada algoritmo antes de treinar.")

with st.sidebar.expander("Isolation Forest", expanded=False):
    if_n_estimators = st.slider(
        "Número de árvores",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        key="if_n_estimators"
    )
    if_contamination = st.slider(
        "Taxa de contaminação (%)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="if_contamination"
    )

with st.sidebar.expander("One-Class SVM", expanded=False):
    svm_nu = st.slider(
        "Nu",
        min_value=0.01,
        max_value=0.30,
        value=0.01,
        step=0.01,
        key="svm_nu"
    )
    svm_kernel = st.selectbox(
        "Kernel",
        ["rbf", "linear", "poly", "sigmoid"],
        index=0,
        key="svm_kernel"
    )
    svm_gamma = st.selectbox(
        "Gamma",
        ["scale", "auto"],
        index=0,
        key="svm_gamma"
    )

with st.sidebar.expander("Local Outlier Factor", expanded=False):
    lof_n_neighbors = st.slider(
        "Número de vizinhos",
        min_value=5,
        max_value=100,
        value=20,
        step=1,
        key="lof_n_neighbors"
    )
    lof_contamination = st.slider(
        "Taxa de contaminação (%)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="lof_contamination"
    )



# ============================================================================
# 4. CARREGAR DADOS
# ============================================================================

st.markdown("## Carregamento de Dados")

if data_source == "Amostras CSV Processadas":
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv')]
    if not processed_files:
        st.error("[ERRO] Nenhum registo processado encontrado em data/processed/")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_files = st.multiselect(
            "Ficheiros estruturados disponíveis:",
            processed_files,
            default=processed_files[:1] if processed_files else None
        )
    
    if selected_files:
        with st.spinner("A indexar séries temporais..."):
            dfs = []
            for file in selected_files:
                df = pd.read_csv(os.path.join(PROCESSED_DIR, file))
                dfs.append(df)
            
            df_data = pd.concat(dfs, ignore_index=True)
            st.success(f"[OK] Instanciados {len(selected_files)} ficheiro(s) | N total = {len(df_data)}")
            st.session_state.df_data = df_data.copy()
    else:
        st.warning("[AVISO] Selecione pelo menos uma matriz de features.")
        st.stop()

else:
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv') and f.startswith('D')]
    if not raw_files:
        st.error("[ERRO] Nenhum dado bruto localizado em data/raw/")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_raw_files = st.multiselect(
            "Ficheiros brutos disponíveis:",
            raw_files,
            default=raw_files[:1] if raw_files else None
        )
    
    if selected_raw_files:
        with st.spinner("A processar e a extrair features analíticas..."):
            colunas_corretas = ['Timestamp', 'Scenario', 'Temp', 'Hum', 'AccX', 'AccY', 'AccZ', 'SysState', 'SampleCount']
            dfs = []
            
            for file in selected_raw_files:
                try:
                    df_raw = pd.read_csv(os.path.join(RAW_DIR, file), names=colunas_corretas, header=0)
                    df_features = extrair_features(df_raw)
                    dfs.append(df_features)
                    st.info(f"[INFO] {file}: {len(df_features)} vetores calculados.")
                except Exception as e:
                    st.error(f"[ERRO] Falha estrutural no ficheiro {file}: {e}")
            
            if dfs:
                df_data = pd.concat(dfs, ignore_index=True)
                st.success(f"[OK] Total de {len(df_data)} instâncias processadas.")
                st.session_state.df_data = df_data.copy()
            else:
                st.error("[ERRO] Falha crítica: nenhuma matriz foi gerada.")
                st.stop()
    else:
        st.warning("[AVISO] Selecione uma fonte de telemetria bruta.")
        st.stop()

# ============================================================================
# 5. EXPLORAÇÃO DE DADOS
# ============================================================================

st.markdown("## Análise Estatística Exploratória")

tab1, tab2, tab3 = st.tabs(["Sumário", "Estatísticas Descritivas", "Distribuições Densidade"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Volume de Amostras (N)", len(df_data))
    with col2:
        st.metric("Dimensões (Features)", len(df_data.columns))
    with col3:
        st.metric("Alocação de Memória (MB)", round(df_data.memory_usage(deep=True).sum() / 1024**2, 2))
    
    st.dataframe(df_data.head(10), width='stretch')

with tab2:
    st.dataframe(df_data.describe(), width='stretch')

with tab3:
    feature_cols = [col for col in df_data.columns if col not in ['Scenario', 'Timestamp', 'SysState', 'SampleCount']]
    selected_feature = st.selectbox("Feature analítica:", feature_cols)
    
    fig = px.histogram(df_data, x=selected_feature, nbins=30, title=f"Histograma de Frequências: {selected_feature}")
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# 6. PREPARAÇÃO DOS DADOS
# ============================================================================

st.markdown("## Pré-processamento e Particionamento")

feature_cols = [col for col in df_data.columns if col not in ['Scenario', 'Timestamp', 'SysState', 'SampleCount']]
X = df_data[feature_cols].fillna(0).astype(float)

# Normalização padrão académica
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Particionamento sequencial temporal (não aleatório para séries temporais)
split_idx = int(len(X_scaled) * train_test_split_ratio)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Atributos Selecionados", len(feature_cols))
with col2:
    st.metric("Amostras de Treino", len(X_train))
with col3:
    st.metric("Amostras de Teste", len(X_test))

# ============================================================================
# 7. TREINAMENTO DO MODELO
# ============================================================================

st.markdown("## Ajuste e Treino do Modelo Estatístico")

def treinar_modelo_selecionado():
    if selected_model == "IsolationForest":
        model = IsolationForest(
            n_estimators=if_n_estimators,
            contamination=if_contamination / 100,
            random_state=42
        )
    elif selected_model == "OneClassSVM":
        model = OneClassSVM(
            nu=svm_nu,
            kernel=svm_kernel,
            gamma=svm_gamma
        )
    else:  # LOF
        model = LocalOutlierFactor(
            n_neighbors=lof_n_neighbors,
            contamination=lof_contamination / 100,
            novelty=True
        )

    model.fit(X_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train_pred = y_train_pred
    st.session_state.y_test_pred = y_test_pred
    st.session_state.feature_cols = feature_cols
    st.session_state.auto_train_requested = False

    joblib.dump(model, os.path.join(MODELS_DIR, f'{selected_model}_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f'{selected_model}_scaler.pkl'))

    st.success(f"[OK] Modelo {selected_model} convergido e exportado.")

if st.session_state.auto_train_requested:
    with st.spinner(f"A ajustar hiperparâmetros de {selected_model}..."):
        treinar_modelo_selecionado()
else:
    st.info("[INFO] Clique em 'Executar Pipeline Completa' para processar dados e treinar automaticamente os modelos com os parâmetros escolhidos.")

# ============================================================================
# 8. RESULTADOS E MÉTRICAS
# ============================================================================

if hasattr(st.session_state, 'model') and st.session_state.model is not None:
    st.markdown("## Avaliação de Desempenho e Métricas")
    
    y_train_pred = st.session_state.y_train_pred
    y_test_pred = st.session_state.y_test_pred
    
    if y_train_pred is None or y_test_pred is None:
        st.warning("[AVISO] Predições ausentes. Inicie o treino do modelo.")
        st.stop()
    
    y_train_pred_binary = (y_train_pred == -1).astype(int)
    y_test_pred_binary = (y_test_pred == -1).astype(int)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Métricas Globais", "Análise de Variância", "Cronologia de Outliers", "Função de Decisão"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Subconjunto: Treino")
            anomalias_train = np.sum(y_train_pred == -1)
            normais_train = np.sum(y_train_pred == 1)
            pct_anomalias = (anomalias_train / len(y_train_pred)) * 100
            
            st.metric("Outliers Detetados", anomalias_train)
            st.metric("Inliers Confirmados", normais_train)
            st.metric("Racio de Anomalias", f"{pct_anomalias:.2f}%")
        
        with col2:
            st.subheader("Subconjunto: Teste")
            anomalias_test = np.sum(y_test_pred == -1)
            normais_test = np.sum(y_test_pred == 1)
            pct_anomalias_test = (anomalias_test / len(y_test_pred)) * 100
            
            st.metric("Outliers Detetados", anomalias_test)
            st.metric("Inliers Confirmados", normais_test)
            st.metric("Racio de Anomalias", f"{pct_anomalias_test:.2f}%")
    
    with tab2:
        st.markdown("### Análise da Distribuição Contaminante por Atributo")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()
        
        top_features = feature_cols[:4] if len(feature_cols) >= 4 else feature_cols
        
        for idx, feature in enumerate(top_features):
            if idx < len(axes):
                feature_idx = feature_cols.index(feature)
                
                data_normal = X_train[y_train_pred == 1, feature_idx]
                data_anomaly = X_train[y_train_pred == -1, feature_idx]
                
                axes[idx].hist(data_normal, bins=20, alpha=0.6, label='Inliers', color='#00CC96', edgecolor='black')
                axes[idx].hist(data_anomaly, bins=20, alpha=0.6, label='Outliers', color='#FF6692', edgecolor='black')
                
                axes[idx].set_title(f'{feature}', fontweight='bold', fontsize=11)
                axes[idx].set_xlabel('Espaço Normalizado')
                axes[idx].set_ylabel('Frequência Absoluta')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
                
                mean_normal = np.mean(data_normal)
                mean_anomaly = np.mean(data_anomaly)
                axes[idx].axvline(mean_normal, color='#00CC96', linestyle='--', linewidth=2)
                axes[idx].axvline(mean_anomaly, color='#FF6692', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Matriz Comparativa de Parâmetros Populacionais")
        
        comparison_data = []
        for feature in top_features:
            feature_idx = feature_cols.index(feature)
            data_normal = X_train[y_train_pred == 1, feature_idx]
            data_anomaly = X_train[y_train_pred == -1, feature_idx]
            
            comparison_data.append({
                'Atributo': feature,
                'Media_Inliers': f"{np.mean(data_normal):.4f}",
                'Media_Outliers': f"{np.mean(data_anomaly):.4f}",
                'Std_Inliers': f"{np.std(data_normal):.4f}",
                'Std_Outliers': f"{np.std(data_anomaly):.4f}",
                'Delta_Media': f"{abs(np.mean(data_normal) - np.mean(data_anomaly)):.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, width='stretch')
    
    with tab3:
        anomaly_timeline_train = (y_train_pred == -1).astype(int)
        anomaly_timeline_test = (y_test_pred == -1).astype(int)
        
        idx_train = np.arange(len(X_train))
        idx_test = np.arange(len(X_train), len(X_train) + len(X_test))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=idx_train, y=anomaly_timeline_train,
            name='Treino (Insample)', mode='markers',
            marker=dict(size=4, color='#636EFA')
        ))
        fig.add_trace(go.Scatter(
            x=idx_test, y=anomaly_timeline_test,
            name='Teste (Out-of-sample)', mode='markers',
            marker=dict(size=4, color='#FF6692')
        ))
        
        fig.add_vline(
            x=len(X_train) - 0.5, line_dash="dash", line_color="gray",
            annotation_text="Limiar Treino/Teste"
        )
        
        fig.update_layout(
            title='Dispersão Temporal de Eventos Anómalos Detetados',
            xaxis_title='Índice Sequencial Real',
            yaxis_title='Estado de Anomalia (0: Normal, 1: Anómalo)',
            height=400, template='plotly_dark'
        )
        st.plotly_chart(fig, width='stretch')
    
    with tab4:
        st.markdown("### Análise da Função de Decisão e Densidade de Scores")
        try:
            if hasattr(st.session_state.model, 'decision_function'):
                scores_train = st.session_state.model.decision_function(X_train)
            elif hasattr(st.session_state.model, 'score_samples'):
                scores_train = st.session_state.model.score_samples(X_train)
            else:
                scores_train = None
            
            if scores_train is not None:
                fig = go.Figure()
                mask_normal = y_train_pred == 1
                fig.add_trace(go.Histogram(
                    x=scores_train[mask_normal], name='Inliers',
                    opacity=0.7, marker_color='#00CC96'
                ))
                fig.add_trace(go.Histogram(
                    x=scores_train[~mask_normal], name='Outliers',
                    opacity=0.7, marker_color='#FF6692'
                ))
                
                fig.update_layout(
                    title=f'Separação das Classes via Score do Modelo ({selected_model})',
                    xaxis_title='Score Obtido', yaxis_title='Frequência Contagem',
                    barmode='overlay', height=400, template='plotly_dark'
                )
                st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.warning(f"[AVISO] Impossível extrair vetor de scores: {str(e)}")
else:
    st.info("[INFO] Aguardando ajuste de modelo para exibição de métricas.")

# ============================================================================
# 9. VISUALIZAÇÕES AVANÇADAS
# ============================================================================

st.markdown("## Análise Gráfica Multivariada")

if hasattr(st.session_state, 'model') and st.session_state.model is not None:
    feature_cols = st.session_state.feature_cols
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train_pred = st.session_state.y_train_pred
    y_test_pred = st.session_state.y_test_pred
    
    viz_tabs = st.tabs(["Dispersão 2D", "Matriz Correlação", "Análise Boxplot", "Projeção PCA 3D", "Série Térmica", "Série Vibratória"])
    
    with viz_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Eixo X:", feature_cols, key="f1")
        with col2:
            feature2 = st.selectbox("Eixo Y:", feature_cols, key="f2", index=1 if len(feature_cols) > 1 else 0)
        
        idx_f1 = feature_cols.index(feature1)
        idx_f2 = feature_cols.index(feature2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_train[y_train_pred == 1, idx_f1], y=X_train[y_train_pred == 1, idx_f2],
            mode='markers', name='Treino - Inlier', marker=dict(size=6, color='#00CC96', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=X_train[y_train_pred == -1, idx_f1], y=X_train[y_train_pred == -1, idx_f2],
            mode='markers', name='Treino - Outlier', marker=dict(size=8, color='#FF6692', symbol='diamond')
        ))
        fig.update_layout(xaxis_title=feature1, yaxis_title=feature2, height=550, template='plotly_dark')
        st.plotly_chart(fig, width='stretch')
        
    with viz_tabs[1]:
        X_combined = np.vstack([X_train, X_test])
        corr_matrix = np.corrcoef(X_combined.T)
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix, x=feature_cols, y=feature_cols,
            colorscale='RdBu', zmid=0, text=np.round(corr_matrix, 2),
            texttemplate='%{text:.2f}'
        ))
        fig.update_layout(title="Matriz de Correlação Linear (Pearson)", height=600, template='plotly_dark')
        st.plotly_chart(fig, width='stretch')
        
    with viz_tabs[2]:
        selected_features_box = st.multiselect("Atributos para análise de variabilidade:", feature_cols, default=feature_cols[:4])
        if selected_features_box:
            fig, axes = plt.subplots(1, len(selected_features_box), figsize=(15, 4))
            if len(selected_features_box) == 1:
                axes = [axes]
            for i, feat in enumerate(selected_features_box):
                f_idx = feature_cols.index(feat)
                axes[i].boxplot([X_train[y_train_pred == 1, f_idx], X_train[y_train_pred == -1, f_idx]], labels=['Inlier', 'Outlier'])
                axes[i].set_title(feat)
            st.pyplot(fig)

    with viz_tabs[3]:
        pca = PCA(n_components=min(3, len(feature_cols)))
        X_train_pca = pca.fit_transform(X_train)
        if X_train_pca.shape[1] == 3:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=X_train_pca[y_train_pred == 1, 0], y=X_train_pca[y_train_pred == 1, 1], z=X_train_pca[y_train_pred == 1, 2],
                mode='markers', name='Inliers', marker=dict(size=3, color='#00CC96', opacity=0.6)
            ))
            fig.add_trace(go.Scatter3d(
                x=X_train_pca[y_train_pred == -1, 0], y=X_train_pca[y_train_pred == -1, 1], z=X_train_pca[y_train_pred == -1, 2],
                mode='markers', name='Outliers', marker=dict(size=5, color='#FF6692')
            ))
            fig.update_layout(title="Redução Dimensional Espacial (PCA)", height=600, template='plotly_dark')
            st.plotly_chart(fig, width='stretch')

    with viz_tabs[4]:
        temp_features = [col for col in feature_cols if 'Temp' in col or 'Hum' in col]
        if temp_features:
            fig = go.Figure()
            for feat in temp_features:
                fig.add_trace(go.Scatter(y=X_train[:, feature_cols.index(feat)], name=feat, mode='lines'))
            fig.update_layout(title="Variáveis Ambientais Estabilizadas", template='plotly_dark')
            st.plotly_chart(fig, width='stretch')

    with viz_tabs[5]:
        acc_features = [col for col in feature_cols if 'Acc' in col and 'Mean' in col]
        if acc_features:
            fig = go.Figure()
            for feat in acc_features:
                fig.add_trace(go.Scatter(y=X_train[:, feature_cols.index(feat)], name=feat, mode='lines'))
            fig.update_layout(title="Componentes Cinemáticas de Vibração", template='plotly_dark')
            st.plotly_chart(fig, width='stretch')
else:
    st.info("[INFO] Módulos de representação gráfica requerem inicialização do modelo.")

# ============================================================================
# 10. EXPORTAÇÃO E RELATÓRIO
# ============================================================================

st.markdown("## Geração de Relatórios")

if hasattr(st.session_state, 'model') and st.session_state.model is not None:
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    
    with exp_col1:
        if st.button("Exportar Parâmetros Serializados", width='stretch'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            joblib.dump(st.session_state.model, os.path.join(MODELS_DIR, f'{selected_model}_model_{timestamp}.pkl'))
            st.success("[OK] Estados persistidos em pkl.")
            
    with exp_col2:
        if st.button("Gerar Relatório de Desempenho TXT", width='stretch'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(RESULTS_DIR, f'report_{selected_model}_{timestamp}.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"PROTOCOLO EXPERIMENTAL - DETEÇÃO DE ANOMALIAS\nData: {timestamp}\nModelo: {selected_model}")
            st.success(f"[OK] Ficheiro gerado com sucesso.")
            
    with exp_col3:
        X_combined = np.vstack([st.session_state.X_train, st.session_state.X_test])
        y_combined = np.hstack([st.session_state.y_train_pred, st.session_state.y_test_pred])
        df_export = pd.DataFrame(X_combined, columns=st.session_state.feature_cols)
        df_export['Prediction'] = y_combined
        csv_export = df_export.to_csv(index=False)
        st.download_button("Descarregar Base de Dados Anotada (CSV)", data=csv_export, file_name="export_predictions.csv", mime="text/csv")

# ============================================================================
# 11. ANÁLISE DETALHADA DE ANOMALIAS
# ============================================================================

st.markdown("## Diagnóstico Estrutural de Outliers")

if hasattr(st.session_state, 'model') and st.session_state.model is not None and st.session_state.df_data is not None:
    y_all_pred = np.hstack([st.session_state.y_train_pred, st.session_state.y_test_pred])
    total_anomalias = np.sum(y_all_pred == -1)
    total_normais = np.sum(y_all_pred == 1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Volume Absoluto Outliers", total_anomalias)
    col2.metric("Volume Absoluto Inliers", total_normais)
    col3.metric("Racio de Contaminação Global", f"{(total_anomalias / len(y_all_pred) * 100):.2f}%")

# ============================================================================
# 12. DETECÇÃO DE CONCEPT DRIFT (Correção Aplicada)
# ============================================================================

st.markdown("## Monitorização de Concept Drift Estatístico")

if hasattr(st.session_state, 'model') and st.session_state.model is not None and st.session_state.df_data is not None and 'Scenario' in st.session_state.df_data.columns:
    st.markdown("### Teste de Hipóteses entre Distribuições Reais (Kolmogorov-Smirnov)")
    
    try:
        # CORREÇÃO: Limpeza rigorosa de strings nulas e 'Unknown' antes da validação
        df_filtered = st.session_state.df_data.dropna(subset=['Scenario'])
        df_filtered = df_filtered[df_filtered['Scenario'].astype(str).str.lower() != 'unknown']
        
        scenarios = sorted(df_filtered['Scenario'].unique())
        
        # Validação do número mínimo de cenários reais para o teste estatístico
        if len(scenarios) > 1:
            if hasattr(st.session_state.model, 'decision_function'):
                scores_all = st.session_state.model.decision_function(X_scaled)
            elif hasattr(st.session_state.model, 'score_samples'):
                scores_all = st.session_state.model.score_samples(X_scaled)
            else:
                scores_all = None
            
            if scores_all is not None:
                df_filtered['Anomaly_Score'] = scores_all[df_filtered.index]
                
                drift_results = []
                for i, scenario_a in enumerate(scenarios):
                    for scenario_b in scenarios[i+1:]:
                        scores_a = df_filtered[df_filtered['Scenario'] == scenario_a]['Anomaly_Score'].values
                        scores_b = df_filtered[df_filtered['Scenario'] == scenario_b]['Anomaly_Score'].values
                        
                        # Garantia de tamanho amostral mínimo para robustez matemática do teste KS
                        if len(scores_a) > 5 and len(scores_b) > 5:
                            ks_stat, p_value = ks_2samp(scores_a, scores_b)
                            drift_detectado = p_value < 0.05
                            
                            drift_results.append({
                                'Cenário A': scenario_a,
                                'Cenário B': scenario_b,
                                'Estatística KS': f"{ks_stat:.4f}",
                                'P-Value': f"{p_value:.6f}",
                                'Modificação Identificada': "SIM (p < 0.05)" if drift_detectado else "NÃO"
                            })
                
                if drift_results:
                    drift_df = pd.DataFrame(drift_results)
                    st.dataframe(drift_df, width='stretch')
                    
                    st.markdown("""
                    **Fundamentação Teórica do Teste Kolmogorov-Smirnov (KS-Test):**
                    * **Estatística KS:** Mede a divergência máxima assintótica entre as distribuições cumulativas empíricas de dois cenários observados.
                    * **P-Value:** Critério de rejeição da hipótese nula ($H_0$: as distribuições são estatisticamente idênticas). Um p-value inferior ao nível de significância de $alpha = 0.05$ denota a presença inequívoca de **Concept Drift**, implicando que a dinâmica subjacente do sistema sofreu mutações estruturais e requer uma atualização ou retreino do classificador preditivo.
                    """)
                else:
                    st.info("[INFO] Densidade amostral insuficiente nos cenários para inferência estatística válida.")
            else:
                st.warning("[AVISO] O algoritmo selecionado impossibilita a extração de funções contínuas de decisão.")
        else:
            st.info("[INFO] Análise suspensa: A base de dados real contém apenas 1 cenário legítimo após a filtragem de valores nulos ou inconclusivos ('Unknown').")
            
    except Exception as e:
        st.warning(f"[AVISO] Erro na engine estatística de drift: {str(e)}")
else:
    st.info("[INFO] Inicialize o treino para permitir os testes estatísticos não-paramétricos de drift.")

# ============================================================================
# 13. FOOTER 
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-family: sans-serif;'>
    <p style='font-size: 13px; color: #aaa; margin-bottom: 4px;'>
        <b>DriftSense-PM</b> | Framework para Deteção de Anomalias e Monitorização de Concept Drift em Tempo Real
    </p>
    <p style='font-size: 12px; color: #888;'>
        Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães
    </p>
</div>
""", unsafe_allow_html=True)
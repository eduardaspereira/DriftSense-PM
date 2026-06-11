"""
DriftSense-PM: Orquestrador de Campanha Full-Factorial Adaptativa
Cumpre rigorosamente os requisitos das Semanas 12, 13 e 14 do Plano Técnico.
"""

import os
import time
import json
import yaml
import numpy as np
import pandas as pd
import psutil
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

def load_config(path="../configs/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

class DriftSenseCampaign:
    def __init__(self, cfg):
        self.cfg = cfg
        # Correção dos caminhos com base na estrutura real do teu config.yaml
        self.features = cfg['feature_engineering']['features']
        self.n_execs = cfg['experiment']['n_executions']
        np.random.seed(cfg['experiment']['random_seed'])
        
    def prepare_scenario(self, path_normal, path_drift):
        df_normal = pd.read_csv(path_normal)[self.features]
        df_drift = pd.read_csv(path_drift)[self.features]
        
        # 30% nominal para treino de base
        split_limit = int(len(df_normal) * 0.3)
        X_train_raw = df_normal.iloc[:split_limit].values
        
        # Stream contínuo (Resto do normal + Falha)
        stream_normal = df_normal.iloc[split_limit:].values
        stream_drift = df_drift.values
        ponto_drift = len(stream_normal)
        
        stream_raw = np.vstack((stream_normal, stream_drift))
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        stream = scaler.transform(stream_raw)
        
        return X_train, stream, ponto_drift

    def run_evaluation(self, scenario_name, path_normal, path_drift):
        X_train, stream, ponto_drift = self.prepare_scenario(path_normal, path_drift)
        centroide_replay = np.mean(X_train, axis=0)
        baseline_rms = X_train[:, 1]
        
        # Buffers para análise estatística das iterações (Exigência da Semana 14)
        latencias_inf = []
        tempos_retreino = []
        fp_count = 0
        delay_drift = None
        gate_triggered = False
        
        print(f"\n[CAMPANHA] A avaliar Cenário {scenario_name} ao longo de {self.n_execs} execuções...")
        
        # Loop de Repetições para garantir o Rigor Estatístico (IC 95%)
        for run in range(self.n_execs):
            ocsvm = OneClassSVM(**self.cfg['models']['oc_svm'])
            ocsvm.fit(X_train)
            
            erros_consecutivos = 0
            acionou_falha = False
            
            # Simulação do fluxo de dados temporal
            t0_stream = time.perf_counter()
            for idx, janela in enumerate(stream):
                is_fase_normal = idx < ponto_drift
                primeira_Janela_divergencia = None

                #  DET1 
                if not acionou_falha:
                    pred = ocsvm.predict([janela])[0]
                    if pred == -1:
                        erros_consecutivos += 1
                        # REGISTO REAL DA PRIMEIRA VEZ QUE O MODELO DETETOU A FALHA
                        if erros_consecutivos == 1 and not is_fase_normal:
                            primeira_janela_divergencia = idx - ponto_drift
                    else:
                        erros_consecutivos = 0
                        
                    if erros_consecutivos >= self.cfg['detectors']['det1_error_monitoring']['persistence']:
                        if is_fase_normal:
                            if run == 0: fp_count += 1 
                            erros_consecutivos = 0
                            primeira_janela_divergencia = None
                        else:
                            distancia = np.linalg.norm(janela - centroide_replay)
                            if distancia > self.cfg['gate']['limite_distancia']:
                                gate_triggered = True
                            
                            # AGORA REPORTAS O ATRASO DO ISOLAMENTO FÍSICO REAL, NÃO DA REGRA FIXA
                            delay_drift = primeira_janela_divergencia if primeira_janela_divergencia is not None else (idx - ponto_drift)
                            acionou_falha = True
                            
                            # Medição do tempo de Re-treino Incremental Híbrido (A2)
                            tamanho_replay = int(len(X_train) * self.cfg['adaptation']['percentage_replay'])
                            indices = np.random.choice(len(X_train), size=tamanho_replay, replace=False)
                            X_hibrido = np.vstack((X_train[indices], stream[ponto_drift:ponto_drift+200]))
                            
                            t0_ret = time.perf_counter()
                            ocsvm.fit(X_hibrido)
                            t1_ret = time.perf_counter()
                            tempos_retreino.append((t1_ret - t0_ret) * 1000)
            
            t1_stream = time.perf_counter()
            latencias_inf.append(((t1_stream - t0_stream) / len(stream)) * 1000)
            
        # Geração de Intervalos de Confiança de 95%
        lat_mean, lat_std = np.mean(latencias_inf), np.std(latencias_inf, ddof=1)
        lat_ci = 1.96 * (lat_std / np.sqrt(self.n_execs))
        
        ret_mean, ret_std = np.mean(tempos_retreino), np.std(tempos_retreino, ddof=1)
        ret_ci = 1.96 * (ret_std / np.sqrt(self.n_execs))
        
        far = (fp_count / ponto_drift) * 100
        
        resultados = {
            "cenario": scenario_name,
            "falsos_alarmes": fp_count,
            "far_percentagem": round(far, 4),
            "atraso_janelas": delay_drift,
            "gate_bloqueio_catastrofico": gate_triggered,
            "latencia_inf_media_ms": round(lat_mean, 5),
            "latencia_inf_ic95_ms": round(lat_ci, 5),
            "retreino_a2_medio_ms": round(ret_mean, 2),
            "retreino_a2_ic95_ms": round(ret_ci, 2)
        }
        
        return resultados

def main():
    cfg = load_config()
    campaign = DriftSenseCampaign(cfg)
    
    # Caminho absoluto para a pasta onde tens os 9 ficheiros processados da imagem
    base_dir = "/home/user/projeto/DriftSense-PM/data/CWRU_dataset/processed/"
    path_normal = os.path.join(base_dir, "Time_Normal_1_098_features.csv")
    
    # Dicionário atualizado com a totalidade dos teus ficheiros reais da CWRU
    cenarios_para_correr = {
        # Família da Pista Interna (Inner Race)
        "IR_007_Ligeiro":      (path_normal, os.path.join(base_dir, "IR007_1_110_features.csv")),
        "IR_014_Medio":        (path_normal, os.path.join(base_dir, "IR014_1_175_features.csv")),
        "IR_021_Catastrofico": (path_normal, os.path.join(base_dir, "IR021_1_214_features.csv")),
        
        # Família da Pista Externa (Outer Race)
        "OR_007_Ligeiro":      (path_normal, os.path.join(base_dir, "OR007_6_1_136_features.csv")),
        "OR_014_Medio":        (path_normal, os.path.join(base_dir, "OR014_6_1_202_features.csv")),
        "OR_021_Catastrofico": (path_normal, os.path.join(base_dir, "OR021_6_1_239_features.csv")),
        
        # Família das Esferas (Ball)
        "B_007_Ligeiro":       (path_normal, os.path.join(base_dir, "B007_1_123_features.csv")),
        "B_014_Medio":         (path_normal, os.path.join(base_dir, "B014_1_190_features.csv")),
        "B_021_Catastrofico":  (path_normal, os.path.join(base_dir, "B021_1_227_features.csv"))
    }
    
    output_final = []
    for nome_cenario, caminhos in cenarios_para_correr.items():
        if not os.path.exists(caminhos[1]):
            print(f"[AVISO] Ficheiro não encontrado para {nome_cenario}. A saltar...")
            continue
            
        res = campaign.run_evaluation(nome_cenario, caminhos[0], caminhos[1])
        output_final.append(res)
        print(f"-> {nome_cenario} Concluído. FAR: {res['far_percentagem']}% | Atraso: {res['atraso_janelas']} janelas.")
        print(f"-> Gate Ativada (Bloqueio): {res['gate_bloqueio_catastrofico']}\n")
        
    os.makedirs("../results/metrics/CWRU_dataset/", exist_ok=True)
    with open("../results/metrics/CWRU_dataset/campanha_factorial_raw.json", "w") as f:
        json.dump(output_final, f, indent=4)
    print("\n[!] Campanha CWRU concluída com sucesso. Ficheiro JSON gerado.")

if __name__ == "__main__":
    main()
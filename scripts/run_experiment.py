import serial
import time
import re
import csv
import os
import yaml
from datetime import datetime
from gpiozero import LED, Button, Motor
from colorama import Fore, Style, init

init(autoreset=True)    

# --- CARREGAR CONFIGURAÇÃO ---
CONFIG_PATH = "../configs/config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {
        'experiment': {'scenario_id': 'RAW_DATA'},
        'system': {'dataset_version': 'v1', 'serial_port': '/dev/ttyACM0', 'baud_rate': 115200},
        'paths': {'raw_data_dir': '../data/raw'}
    }

SCENARIO = config['experiment']['scenario_id']
VERSION = config['system']['dataset_version']
RAW_DIR = config['paths']['raw_data_dir']
CSV_FILENAME = os.path.join(RAW_DIR, f"dataset_{SCENARIO}_{VERSION}_raw.csv")

# --- GPIO ---
led_green, led_red = LED(17, active_high=False), LED(27, active_high=False)
fan = Motor(forward=24, backward=23)
button = Button(22, pull_up=False)

# --- ESTADO ---
system_enabled = False
sample_count = 0

def toggle_system():
    global system_enabled
    system_enabled = not system_enabled
    if system_enabled:
        led_green.on(); led_red.off()
        print(f"\n{Fore.GREEN}[SISTEMA ATIVADO] Ventoinha ligada a 50%. A gravar raw a 2Hz...{Style.RESET_ALL}")
    else:
        led_green.off(); led_red.on()
        fan.stop()
        print(f"\n{Fore.RED}[SISTEMA DESATIVADO] Ventoinha parada.{Style.RESET_ALL}")

button.when_pressed = toggle_system

def main():
    global sample_count
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # --- Cronómetro para os 0.5 segundos ---
    ultimo_tempo_gravacao = time.time() 

    try:
        ser = serial.Serial(config['system']['serial_port'], config['system']['baud_rate'], timeout=0.1)
        
        with open(CSV_FILENAME, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Timestamp", "Scenario", "Temp", "Hum", "AccX", "AccY", "AccZ", "Sample"])

            print(f"{Fore.CYAN}{Style.BRIGHT}=== DriftSense: RAW a 2Hz (Ventoinha Controlada Apenas Pelo Botão) ===")
            print("-" * 105)
            print(f"{'TIMESTAMP':<10} | {'TEMP':<5} | {'HUM':<5} | {'ACC_X':<7} | {'ACC_Y':<7} | {'ACC_Z':<7} | {'SAMPLE'}")

            temp, hum = 0.0, 0.0

            while True:
                if ser.in_waiting > 0:
                    raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_line)

                    # 1. Atualizar Temperatura/Humidade
                    if "TEMP" in raw_line.upper() and len(nums) >= 2:
                        temp, hum = float(nums[0]), float(nums[1])

                    # 2. Processar Vibração apenas a cada 0.5s
                    elif "ACCEL" in raw_line.upper() and len(nums) >= 3:
                        tempo_atual = time.time()
                        
                        # Só entra se a diferença for >= 0.5 segundos
                        if (tempo_atual - ultimo_tempo_gravacao) >= 0.5:
                            ultimo_tempo_gravacao = tempo_atual 
                            
                            ax, ay, az = float(nums[0]), float(nums[1]), float(nums[2])
                            ts = datetime.now().strftime("%H:%M:%S")
                            sys_state = 1 if system_enabled else 0
                            sample_count += 1
                            
                            # Gravação imediata da amostra pura
                            writer.writerow([ts, SCENARIO, temp, hum, ax, ay, az, sys_state, sample_count])
                            
                            # Feedback no terminal - FAN depende APENAS do estado do sistema
                            #fan_status = "50%" if system_enabled else "OFF"
                            color = Fore.GREEN if system_enabled else Fore.WHITE
                            print(f"{ts:<10} | {temp:<5.1f} | {hum:<5.1f} | {ax:<7.2f} | {ay:<7.2f} | {az:<7.2f} | {Fore.YELLOW}{sample_count}")
                            
                            # Forçar escrita no disco
                            csv_file.flush()

                # --- CONTROLO DA VENTOINHA (Ignora Temperatura) ---
                if system_enabled:
                    fan.forward(0.5)
                else:
                    fan.stop()

                time.sleep(0.001)

    except Exception as e:
        print(f"\n{Fore.RED}Erro: {e}")
    finally:
        fan.stop()
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    main()
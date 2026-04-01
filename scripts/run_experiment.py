import serial
import time
import re
import csv
import os
import yaml
import numpy as np
from datetime import datetime
from gpiozero import LED, Button
from collections import deque
from colorama import Fore, Style, init

init(autoreset=True)

# --- CARREGAR CONFIGURAÇÃO ---
CONFIG_PATH = "../configs/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

SCENARIO = config['experiment']['scenario_id']
VERSION = config['system']['dataset_version']
WINDOW_SIZE = config['experiment']['window_size'] # 100
RAW_DIR = config['paths']['raw_data_dir']
CSV_FILENAME = os.path.join(RAW_DIR, f"dataset_{SCENARIO}_{VERSION}.csv")

# --- GPIO ---
green_led, red_led = LED(17), LED(27)
motor_BI, motor_FI = LED(23), LED(24)
button = Button(22)

# --- ESTADO ---
system_enabled = False
accel_buffer = deque(maxlen=WINDOW_SIZE)
window_count = 0

def calculate_rms(buffer):
    """Calcula RMS (Root Mean Square)."""
    data = np.array(buffer, dtype=float)
    return np.sqrt(np.mean(data**2, axis=0))

def toggle_system():
    global system_enabled
    system_enabled = not system_enabled
    green_led.value = system_enabled
    red_led.value = not system_enabled


button.when_pressed = toggle_system

def main():
    global window_count
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Validação de integridade do dataset
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, 'r') as f:
            window_count = max(0, sum(1 for _ in f) - 1)

    try:
        ser = serial.Serial(config['system']['serial_port'], config['system']['baud_rate'], timeout=0.1)
        
        with open(CSV_FILENAME, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if window_count == 0:
                # Cabeçalho do CSV atualizado
                writer.writerow(["Timestamp", "Scenario", "Temp", "Hum", "AccX_RMS", "AccY_RMS", "AccZ_RMS", "SysActive", "WindowCount"])

            print(f"{Fore.WHITE}{Style.BRIGHT}Monitor DriftSense-PM: {SCENARIO}")
            print(f"A aguardar preenchimento de janelas ({WINDOW_SIZE} amostras)...")
            print("-" * 105)
            
            # Cabeçalho do Terminal atualizado com SYS e COUNT
            print(f"{'TIMESTAMP':<10} | {'SCEN':<5} | {'TEMP':<5} | {'HUM':<5} | {'RMS_X':<7} | {'RMS_Y':<7} | {'RMS_Z':<7} | {'SYS':<3} | {'COUNT'}")

            temp, hum = 0.0, 0.0

            while True:
                if ser.in_waiting > 0:
                    raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_line)

                    if "TEMP" in raw_line.upper() and len(nums) >= 2:
                        temp, hum = float(nums[0]), float(nums[1])

                    # Mantive apenas "VIBRA" para contornar problemas com o "Ç" e o "Ã"
                    elif "VIBRA" in raw_line.upper() and len(nums) >= 3:
                        accel_buffer.append([float(nums[0]), float(nums[1]), float(nums[2])])

                        # Retirada a condição 'system_enabled' daqui para gravar sempre (com a flag 0 ou 1)
                        if len(accel_buffer) == WINDOW_SIZE:
                            rms = calculate_rms(accel_buffer)
                            ts = datetime.now().strftime("%H:%M:%S")
                            
                            # Transforma o estado booleano num inteiro (1 para True, 0 para False)
                            sys_state = 1 if system_enabled else 0
                            
                            window_count += 1
                            
                            # Grava no ficheiro CSV com as novas colunas
                            writer.writerow([ts, SCENARIO, temp, hum, f"{rms[0]:.2f}", f"{rms[1]:.2f}", f"{rms[2]:.2f}", sys_state, window_count])
                            csv_file.flush()
                            
                            # IMPRESSÃO NO TERMINAL com o estado 0/1 e a contagem da janela
                            print(f"{ts:<10} | {SCENARIO:<5} | {temp:<5.1f} | {hum:<5.1f} | {rms[0]:<7.2f} | {rms[1]:<7.2f} | {rms[2]:<7.2f} | {Fore.GREEN if sys_state else Fore.RED}{sys_state:<3}{Style.RESET_ALL} | {Fore.YELLOW}{window_count}")

                            # Limpa o buffer para iniciar nova janela
                            accel_buffer.clear()

                # Controlo de Hardware mantém-se igual (só liga motores se estiver ATIVO)
                if system_enabled and temp > config['system']['temp_threshold']:
                    motor_BI.on(); motor_FI.off()
                else:
                    motor_BI.off(); motor_FI.off()

                time.sleep(0.001)

    except Exception as e:
        print(f"\n{Fore.RED}Erro: {e}")
    finally:
        motor_BI.off(); motor_FI.off()
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    main()
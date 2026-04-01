import serial
import time
import re
import csv
import os
from datetime import datetime
from gpiozero import LED, Button

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
CSV_FILENAME = "../data/raw/portenta_data_log.csv"
TEMP_THRESHOLD = 23.0

# --- GPIO SETUP ---
green_led = LED(17)
red_led = LED(27)
motor_BI = LED(23)
motor_FI = LED(24)
button = Button(22)

# --- SYSTEM STATE ---
system_enabled = False
current_temp = 0.0
# Variáveis persistentes para os sensores (do file1.py)
temp_val, hum_val, gas_val = "0.00", "0.00", "0.00"
acc_x, acc_y, acc_z = "0", "0", "0"

# --- FUNCTIONS ---

def fan_on():
    motor_BI.on()
    motor_FI.off()

def fan_off():
    motor_BI.off()
    motor_FI.off()

def toggle_system():
    """Alterna o estado do sistema e LEDs de feedback."""
    global system_enabled
    system_enabled = not system_enabled
    
    if system_enabled:
        print("\n" + "="*20)
        print("[SISTEMA ATIVADO] Monitorização e Controlo ON")
        print("="*20)
        green_led.on()
        red_led.off()
    else:
        print("\n" + "="*20)
        print("[SISTEMA DESATIVADO] Fan em OFF forçado")
        print("="*20)
        green_led.off()
        red_led.on()
        fan_off()

# Interrupção do botão
button.when_pressed = toggle_system

def main():
    global current_temp, temp_val, hum_val, gas_val, acc_x, acc_y, acc_z
    
    # Garante que a pasta existe e verifica se o ficheiro já existe
    os.makedirs(os.path.dirname(CSV_FILENAME), exist_ok=True)
    file_exists = os.path.isfile(CSV_FILENAME)

    try:
        # Inicializa Conexão Serial
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2) 
        
        # Cabeçalho Inicial no Terminal
        print("\n" + "="*85)
        print(f"SISTEMA INTEGRADO - ACADEMIC LOGGING & FAN CONTROL")
        print(f"CSV: {os.path.abspath(CSV_FILENAME)}")
        print(f"Threshold: {TEMP_THRESHOLD}°C")
        print("="*85)

        # Cabeçalho das Colunas (Estilo file1.py)
        header_text = f"{'TIMESTAMP':<20} | {'TEMP (C)':<8} | {'HUM (%)':<8} | {'GAS (Ohm)':<10} | {'ACCEL (X,Y,Z)'}"
        print(header_text)
        print("-" * len(header_text))

        # Estado Inicial de Hardware
        green_led.off()
        red_led.on()
        fan_off()

        # Abrir CSV para escrita
        with open(CSV_FILENAME, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Escreve cabeçalho se o ficheiro for novo
            if not file_exists:
                writer.writerow(["Timestamp", "Temperature_C", "Humidity_Pct", "Gas_Resistance_Ohm", "Accel_X", "Accel_Y", "Accel_Z"])

            while True:
                if ser.in_waiting > 0:
                    raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_line)

                    # 1. Atualizar Variáveis de Ambiente
                    if "TEMP" in raw_line.upper() and len(nums) >= 2:
                        temp_val, hum_val = nums[0], nums[1]
                        current_temp = float(temp_val) # Usado para a lógica da fan

                    # 2. Atualizar Acelerómetro
                    elif "ACCEL" in raw_line.upper() and len(nums) >= 3:
                        acc_x, acc_y, acc_z = nums[0], nums[1], nums[2]
                    
                    # 3. Atualizar Gás e disparar a Impressão/Escrita (Sincronizado)
                    elif "GAS" in raw_line.upper() and len(nums) >= 1:
                        gas_val = nums[0]
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Imprimir no Terminal (Colunas do file1.py)
                        accel_str = f"({acc_x}, {acc_y}, {acc_z})"
                        print(f"{ts:<20} | {temp_val:<8} | {hum_val:<8} | {gas_val:<10} | {accel_str}")

                        # Salvar no CSV
                        writer.writerow([ts, temp_val, hum_val, gas_val, acc_x, acc_y, acc_z])
                        csv_file.flush() 

                # --- LÓGICA DE CONTROLO DA VENTOINHA ---
                if system_enabled:
                    if current_temp > TEMP_THRESHOLD:
                        fan_on()
                    else:
                        fan_off()
                else:
                    fan_off()

                time.sleep(0.05) # Evita spike de CPU

    except serial.SerialException as e:
        print(f"\nERRO DE CONEXÃO: {e}")
    except KeyboardInterrupt:
        print("\nA encerrar e limpar GPIO...")
    finally:
        fan_off()
        green_led.off()
        red_led.off()
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
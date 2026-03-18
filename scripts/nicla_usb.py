import serial
import time
import re
import csv
import os
from datetime import datetime

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
CSV_FILENAME = "../data/raw/nicla_data_log.csv"

def run_acquisition():
    # Check if file exists to decide whether to write the header
    file_exists = os.path.isfile(CSV_FILENAME)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) 
        
        print("\n" + "="*85)
        print(f"NICLA SENSE ME - DATA ACQUISITION & CSV LOGGING")
        print(f"Logging to: {os.path.abspath(CSV_FILENAME)}")
        print("="*85)

        # Terminal Header
        header_text = f"{'TIMESTAMP':<20} | {'TEMP (C)':<8} | {'HUM (%)':<8} | {'GAS (Ohm)':<10} | {'ACCEL (X,Y,Z)'}"
        print(header_text)
        print("-" * len(header_text))

        # Sensor State Variables
        temp, hum, gas = "0.00", "0.00", "0.00"
        acc_x, acc_y, acc_z = "0", "0", "0"

        # Open the CSV file in 'append' mode
        with open(CSV_FILENAME, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header only if the file is new
            if not file_exists:
                writer.writerow(["Timestamp", "Temperature_C", "Humidity_Pct", "Gas_Resistance_Ohm", "Accel_X", "Accel_Y", "Accel_Z"])

            while True:
                if ser.in_waiting > 0:
                    raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_line)

                    # 1. Parse Environmental Data
                    if "TEMP" in raw_line.upper() and len(nums) >= 2:
                        temp, hum = nums[0], nums[1]

                    # 2. Parse Acceleration (X, Y, Z)
                    elif len(nums) == 3:
                        acc_x, acc_y, acc_z = nums[0], nums[1], nums[2]
                    
                    # 3. Parse Gas & Trigger Row Save
                    elif "GAS" in raw_line.upper() and len(nums) >= 1:
                        gas = nums[0]
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Print to Terminal
                        accel_str = f"({acc_x}, {acc_y}, {acc_z})"
                        print(f"{ts:<20} | {temp:<8} | {hum:<8} | {gas:<10} | {accel_str}")

                        # Save to CSV
                        writer.writerow([ts, temp, hum, gas, acc_x, acc_y, acc_z])
                        csv_file.flush() # Forces data to be written to disk immediately

    except serial.SerialException as e:
        print(f"\nCRITICAL ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\nLogging stopped. File saved successfully.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    run_acquisition()
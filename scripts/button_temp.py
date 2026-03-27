import serial
import time
import re
from gpiozero import LED, Button

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'  # Adjust if your Portenta is on a different port
BAUD_RATE = 115200
TEMP_THRESHOLD = 23.0

# --- GPIO SETUP ---
# LEDs
green_led = LED(27)
red_led = LED(17)

# Motor driver
motor_BI = LED(23)
motor_FI = LED(24)

# Button
button = Button(25)

# --- SYSTEM STATE ---
system_enabled = False
current_temp = 0.0

# --- FUNCTIONS ---

def fan_on():
    motor_BI.on()
    motor_FI.off()

def fan_off():
    motor_BI.off()
    motor_FI.off()

def toggle_system():
    """Toggles whether the automatic fan control is active."""
    global system_enabled
    system_enabled = not system_enabled
    
    if system_enabled:
        print("\n[SYSTEM ENABLED] Monitoring temperature...")
        green_led.on()
        red_led.off()
    else:
        print("\n[SYSTEM DISABLED] Fan forced OFF.")
        green_led.off()
        red_led.on()
        fan_off()

# Attach button interrupt
button.when_pressed = toggle_system

def main():
    global current_temp
    
    try:
        # Initialize Serial connection to Portenta C33
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2) # Wait for Portenta to initialize
        
        # Initial State
        green_led.off()
        red_led.on()
        fan_off()
        
        print("="*50)
        print("PORTENTA C33 + NICLA SENSE FAN CONTROL")
        print(f"Threshold: {TEMP_THRESHOLD}ï¿½C | Port: {SERIAL_PORT}")
        print("="*50)

        while True:
            # 1. Read Serial Data from Portenta
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Use regex from portenta.py to extract numbers
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_line)

                # Update temperature if the line contains "TEMP"
                if "TEMP" in raw_line.upper() and len(nums) >= 1:
                    current_temp = float(nums[0])
                    status = "ACTIVE" if system_enabled else "IDLE"
                    print(f"[{status}] Temp: {current_temp:.2f}ï¿½C", end='\r')

            # 2. Logic Control
            if system_enabled:
                # Automatic fan control based on Nicla sensor data
                if current_temp > TEMP_THRESHOLD:
                    fan_on()
                else:
                    fan_off()
            else:
                # Ensure fan is off when system is disabled
                fan_off()

            # Small sleep to prevent CPU spiking, 
            # but short enough to keep serial buffer clear
            time.sleep(0.1)

    except serial.SerialException as e:
        print(f"\nCONNECTION ERROR: {e}")
    except KeyboardInterrupt:
        print("\nExiting and cleaning up...")
    finally:
        fan_off()
        green_led.off()
        red_led.off()
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
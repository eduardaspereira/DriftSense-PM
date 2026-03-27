from gpiozero import Motor, LED, Button
from signal import pause

# 1. Configuraï¿½ï¿½o do Hardware
# LEDs: Lï¿½gica Invertida (Active Low)
led_green = LED(17, active_high=False) 
led_red   = LED(27, active_high=False)

# BOTï¿½O: Mantendo a tua lï¿½gica que funcionou (active_state=False)
button = Button(25, pull_up=None, active_state=False)

# Motor: FI=24, BI=23
fan = Motor(forward=24, backward=23)

# 2. Funï¿½ï¿½o para Alternar (Toggle)
def alternar_estado():
    if fan.is_active:
        # Se estiver a girar, para
        print("Botï¿½o detetado: A parar a ventoinha...")
        fan.stop()
        led_green.off()
        led_red.on()
    else:
        # Se estiver parada, liga
        print("Botï¿½o detetado: A ligar a ventoinha!")
        fan.forward()
        led_green.on()
        led_red.off()

# 3. Estado Inicial (Sistema arranca parado)
print("Sistema Pronto. Toca no botï¿½o para ligar/desligar.")
fan.stop()
led_green.off()
led_red.on()

# 4. Configuraï¿½ï¿½o do Evento
# Agora apenas detetamos o clique inicial para alternar o estado
button.when_pressed = alternar_estado

# Mantï¿½m o script em execuï¿½ï¿½o
pause()
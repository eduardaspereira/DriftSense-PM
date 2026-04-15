from gpiozero import Motor, LED, Button
from signal import pause

# 1. Configuração do Hardware
led_green = LED(17, active_high=False) 
led_red   = LED(27, active_high=False)

# BOTÃO: Mantendo a tua lógica que funcionou
button = Button(22, pull_up=False)

# Motor: FI=24, BI=23
fan = Motor(forward=24, backward=23)

# 2. Função para Alternar (Toggle)
def alternar_estado():
    if fan.is_active:
        # Se estiver a girar, para
        print("Botão detetado: A parar a ventoinha...")
        fan.stop()
        led_green.off()
        led_red.on()
    else:
        # Se estiver parada, liga a 50% da velocidade
        print("Botão detetado: A ligar a ventoinha a 50%!")
        fan.forward(0.5) # <--- A MAGIA ACONTECE AQUI
        led_green.on()
        led_red.off()

# 3. Estado Inicial (Sistema arranca parado)
print("Sistema Pronto. Toca no botão para ligar/desligar.")
fan.stop()
led_green.off()
led_red.on()

# 4. Configuração do Evento
# Agora apenas detetamos o clique inicial para alternar o estado
button.when_pressed = alternar_estado

# Mantém o script em execução
pause()
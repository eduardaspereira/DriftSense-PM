#!/usr/bin/env python3
"""
Descrição: Logger para o medidor de energia FNIRSI-FNB58 (versão cross-OS).
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Este módulo fornece uma interface para capturar leituras do medidor FNIRSI
e guardar num CSV. Adaptações foram feitas para suportar Windows, Linux e Mac
quando possível.

Uso (Windows):
    python power_meter_fnirsi_windows.py --output power_measurements.csv --duration 11400

Requisitos:
    - pyusb
    - pandas
    - numpy
"""

import sys
import time
import csv
import argparse
import signal
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import usb.core
    import usb.util
except ImportError:
    print("Erro: pyusb não instalado.")
    if platform.system() == "Windows":
        print("   Execute no PowerShell (como Administrator):")
        print("   pip install pyusb")
    else:
        print("   Execute: pip install pyusb")
    sys.exit(1)


# --- FNIRSI Device IDs ---
VID_FNB58 = 0x2E3C
PID_FNB58 = 0x5558

# Legacy models
VID = 0x0483
PID_FNB48 = 0x003A
PID_C1 = 0x003B

VID_FNB48S = 0x2E3C
PID_FNB48S = 0x0049


class FNIRSIPowerMeter:
    """Interface para comunicação com FNIRSI Power Meter (Windows-friendly)"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device = None
        self.is_fnb58_or_fnb48s = False
        self.ep_in = None
        self.ep_out = None
        self.start_time = None
        self.energy_accumulated = 0.0
        self.capacity_accumulated = 0.0
        self.sample_count = 0
        self.sps = 100  # Samples per second
        self.time_interval = 1.0 / self.sps
        self.os_name = platform.system()
        
    def connect(self):
        """Conectar ao dispositivo"""
        self.device, self.is_fnb58_or_fnb48s = self._find_device()
        
        if not self.device:
            raise RuntimeError(
                f"Dispositivo FNIRSI não encontrado!\n"
                f"   Verifique a ligação USB.\n"
                f"   Em Windows: Abra Gestor de Dispositivos e procure por 'FNIRSI' ou 'STMicroelectronics'"
            )
        
        device_type = "FNB58/FNB48S" if self.is_fnb58_or_fnb48s else "FNB48/C1"
        print(f"Dispositivo {device_type} encontrado")
        
        # Em Windows, não é necessário desanexar kernel driver
        # Mas em Linux/Mac, podemos tentar se necessário
        if self.os_name in ["Linux", "Darwin"]:
            self._try_detach_kernel_driver()
        
        try:
            self.device.reset()
            self.device.set_configuration()
        except usb.core.USBError as e:
            if "Resource busy" in str(e):
                raise RuntimeError(
                    f"Dispositivo ocupado (Resource busy).\n"
                    f"   Em Windows: Reinicie o PC ou desconecte/reconecte o USB.\n"
                    f"   Em Linux: Execute: sudo bash cleanup_usb.sh"
                )
            raise
        
        # Encontrar endpoints
        cfg = self.device.get_active_configuration()
        
        # Tentar encontrar interface HID
        interface = None
        for intf in cfg:
            if intf.bInterfaceClass == 0x03:  # HID class
                interface = intf
                break
        
        if not interface:
            # Se não encontrar HID, usar primeira interface
            interface = cfg[(0, 0)]
        
        self.ep_out = usb.util.find_descriptor(
            interface,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) 
                                   == usb.util.ENDPOINT_OUT,
        )
        
        self.ep_in = usb.util.find_descriptor(
            interface,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) 
                                   == usb.util.ENDPOINT_IN,
        )
        
        if not self.ep_in or not self.ep_out:
            raise RuntimeError("Endpoints USB não encontrados")
        
        self._request_data()
        self.start_time = time.time()
        print(f"Comunicação iniciada ({self.os_name}). Começando captura de dados...")
    
    def _try_detach_kernel_driver(self):
        """Tentar desanexar kernel driver (apenas Linux/Mac)"""
        try:
            for cfg in self.device:
                for interface in cfg:
                    if self.device.is_kernel_driver_active(interface.bInterfaceNumber):
                        try:
                            self.device.detach_kernel_driver(interface.bInterfaceNumber)
                            print(f"Kernel driver desanexado")
                        except usb.core.USBError as e:
                            if "not supported" not in str(e).lower():
                                pass  # Ignorar e continuar
        except:
            pass  # Ignorar erros de desanexação
    
    def _find_device(self):
        """Encontrar dispositivo FNIRSI conectado"""
        dev = usb.core.find(idVendor=VID, idProduct=PID_FNB48)
        is_fnb58 = False
        
        if dev is None:
            dev = usb.core.find(idVendor=VID, idProduct=PID_C1)
        if dev is None:
            dev = usb.core.find(idVendor=VID_FNB58, idProduct=PID_FNB58)
            is_fnb58 = True if dev else False
        if dev is None:
            dev = usb.core.find(idVendor=VID_FNB48S, idProduct=PID_FNB48S)
            is_fnb58 = True if dev else False
        
        return dev, is_fnb58
    
    def _request_data(self):
        """Iniciar pedido de dados"""
        self.ep_out.write(b"\xaa\x81" + b"\x00" * 61 + b"\x8e")
        self.ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
        
        if self.is_fnb58_or_fnb48s:
            self.ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
        else:
            self.ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")
    
    def read_samples(self):
        """Ler dados do dispositivo. Retorna lista de dicts com dados."""
        try:
            data = self.ep_in.read(size_or_buffer=64, timeout=5000)
        except usb.core.USBTimeoutError:
            return None
        except usb.core.USBError:
            return None
        
        if data[1] != 0x04:  # Apenas data packets
            return None
        
        samples = []
        elapsed = time.time() - self.start_time
        
        for i in range(4):
            offset = 2 + 15 * i
            
            # Parse 15-byte sample
            voltage = (
                data[offset + 3] * 256 ** 3 +
                data[offset + 2] * 256 ** 2 +
                data[offset + 1] * 256 +
                data[offset + 0]
            ) / 100000
            
            current = (
                data[offset + 7] * 256 ** 3 +
                data[offset + 6] * 256 ** 2 +
                data[offset + 5] * 256 +
                data[offset + 4]
            ) / 100000
            
            temp_c = (data[offset + 13] + data[offset + 14] * 256) / 10.0
            
            # Cálculos acumulativos
            power_w = voltage * current
            self.energy_accumulated += power_w * self.time_interval  # Joules
            self.capacity_accumulated += current * self.time_interval  # Coulombs
            self.sample_count += 1
            
            sample = {
                'timestamp_unix': time.time(),
                'timestamp_iso': datetime.now().isoformat(),
                'voltage_v': voltage,
                'current_a': current,
                'power_w': power_w,
                'temp_c': temp_c,
                'energy_ws': self.energy_accumulated,  # Watt-seconds
                'capacity_as': self.capacity_accumulated,  # Ampere-seconds
                'duration_sec': elapsed
            }
            samples.append(sample)
        
        return samples
    
    def keep_alive(self):
        """Manter comunicação ativa (refresh periódico)"""
        refresh = 1.0 if self.is_fnb58_or_fnb48s else 0.003
        if time.time() - self.start_time > refresh:
            try:
                self.ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")
                self.start_time = time.time()
            except:
                pass
    
    def disconnect(self):
        """Desligar o dispositivo"""
        if self.device:
            try:
                self.device.reset()
            except:
                pass


def save_to_csv(samples, output_file):
    """Guardar amostras em CSV"""
    if not samples:
        return
    
    fieldnames = samples[0].keys()
    
    # Append mode se ficheiro já existe
    file_exists = Path(output_file).exists()
    
    try:
        with open(output_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(samples)
    except IOError as e:
        print(f"Erro ao escrever CSV: {e}")


def print_statistics(meter: FNIRSIPowerMeter):
    """Imprimir estatísticas de consumo"""
    print("\n" + "=" * 70)
    print("ESTATÍSTICAS DE CONSUMO ENERGÉTICO")
    print("=" * 70)
    print(f"Amostras recolhidas: {meter.sample_count:,}")
    print(f"Duração: {meter.sample_count / meter.sps / 60:.1f} min")
    print(f"Energia acumulada: {meter.energy_accumulated:.2f} Ws = {meter.energy_accumulated/3600:.3f} Wh")
    print(f"Capacidade acumulada: {meter.capacity_accumulated:.2f} As")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Capturador FNIRSI-FNB58 para Windows/PC"
    )
    parser.add_argument(
        '--output', '-o',
        default='power_measurements.csv',
        help='Ficheiro CSV de saída (default: power_measurements.csv)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=3600,
        help='Duração máxima em segundos (default: 3600 = 1h)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=5.0,
        help='Intervalo de escrita em CSV em segundos (default: 5)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Modo verbose'
    )
    
    args = parser.parse_args()
    
    meter = FNIRSIPowerMeter(verbose=args.verbose)
    
    try:
        meter.connect()
        
        start_time = time.time()
        last_write_time = start_time
        batch_samples = []
        
        print(f"Intervalo de escrita: {args.interval}s")
        
        while time.time() - start_time < args.duration:
            try:
                samples = meter.read_samples()
                
                if samples:
                    batch_samples.extend(samples)
                    
                    # Escrever em lotes para melhor performance
                    if time.time() - last_write_time > args.interval:
                        save_to_csv(batch_samples, args.output)
                        
                        # Print update
                        latest = samples[-1]
                        print(
                            f"[{latest['timestamp_iso']}] "
                            f"U={latest['voltage_v']:.2f}V | "
                            f"I={latest['current_a']:.3f}A | "
                            f"P={latest['power_w']:.2f}W | "
                            f"T={latest['temp_c']:.1f}°C | "
                            f"E={latest['energy_ws']:.0f}Ws"
                        )
                        
                        batch_samples = []
                        last_write_time = time.time()
                
                meter.keep_alive()
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\n\nCaptura interrompida pelo utilizador")
                break
            except usb.core.USBError as e:
                print(f"Erro USB: {e}")
                print("   Verifique a ligação do power meter e tente novamente")
                break
            except Exception as e:
                print(f"Erro: {e}")
                break
        
        # Escrever dados finais
        if batch_samples:
            save_to_csv(batch_samples, args.output)
        
        print_statistics(meter)
        print(f"Dados guardados em: {Path(args.output).absolute()}")
        
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        meter.disconnect()
        print("Desligado com sucesso")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FNIRSI-FNB58 Power Meter Capture")
    print("=" * 70 + "\n")
    main()

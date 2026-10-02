import time
import sys

from RavenSniffer_windows import *

def mari():
    sniffer = RavenSniffer(data_path,ping_interval=5)


    print(f"inicializando....")
    if not sniffer.load_network():
        print(f"falha ao carregar rede:")
        sys.exit(1)

    print(f"rede carregada....{len(sniffer.hosts)} terminais encontrados.")
    time.sleep(3)

    print(f'mandando pacotes ECHO para {len(sniffer.hosts)} hosts...')

    for host_info in sniffer.hosts:
        host_ip = host_info["ip"]
        host_name = host_info["name"]

        print(f'executando: {host_ip} ({host_name})...')

        latencias = sniffer.pinger(host_ip)
        media = sum(map(int,latencias)) / len(latencias)

        if host_info != sniffer.hosts[-1]:
            time.sleep(sniffer.ping_interval)


        print('protocolo ECHO executado com sucesso.')
if __name__ == '__main__':
    mari()


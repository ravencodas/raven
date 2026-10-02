import subprocess
import platform
import json
from json import JSONDecodeError
import time
from datetime import datetime
import os
import re

#tomar, controlar, superar e jamais recuar


#tempo aqui: (3 + (0,38*60)) + (1 + (0,51*60))
system = platform.system()
tempo = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


#===================== caminhos ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "pops.json")
output_file = os.path.join(base_dir, "output.txt")
rtts_path = os.path.join(base_dir, "rtts.txt")

#=============================================================================================
class RavenSniffer:
    def __init__(self,config_file, ping_interval: int = 5):
        self.config_file = data_path
        self.ping_interval = ping_interval
        self.hosts = []
        self.results = {}
        self.running = False
        self.rtt = []


    def load_network(self): #carrega os dados do json
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                network = json.load(f)

            for dot in network: #a network é o json bruto, o dot é o item "routers"
                for router in dot["routers"]:
                    self.hosts.append({  #cada item da lista v
                        "name": router["name"],
                        "ip": router["ip"]
                    })
            print(f'importação da rede[OK], {len(self.hosts)} carregados.')
            return True
        except FileNotFoundError:
            print(f'Erro ao carregar arquivo {self.config_file}: não encontrado.')
            return False

        except JSONDecodeError as e:
            print(f'ERRO: Arquivo .JSON mal formatado')
            print(f"   Detalhe: {e}")
            print(f"   Linha {e.lineno}, posição {e.colno}")
            return False

        except Exception as e: #elaborar a exceção para manipulação de arquivo json
            print(e)
            return False

        except KeyError as e:
            print(f"ERRO: estrutura do JSON inválida")
            print(f"   Chave ausente: {e}")

        except Exception as e:
            print(f"ERRO: {type(e).__name__} - {e}")
            return False


#========= CÓDIGO 100% OK ATÉ AQUI ==========================================================
    def encontrar_latencia(self, output: str):
        # Padrão para Windows: "tempo=10ms" ou "time=10ms"
        padroes = [
            r'tempo[=\s]*(\d+)ms',
            r'time[=\s]*(\d+)ms',
            r'[=\s](\d+)ms'
        ]

        for padrao in padroes:
            match = re.search(padrao, output, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None  # Retorna None se não encontrar

    def pinger(self,host:str):
        result = subprocess.run(
            ['ping', '-n','4',host],
            capture_output=True,
            text=True,
            timeout = 5

        )
#o host é o endereço de ip!
        if result.returncode == 0:
            raw_output = result.stdout
            linhas = raw_output.splitlines()

            host_info = None
            for h in self.hosts:
                if h['ip'] == host:
                    host_info = h
                    break

            if host_info:
                cabecalho = f"\n=== {host_info['name']} - {host} ===\n"
            else:
                cabecalho = f"\n=== {host} ===\n"

            rtt_host = []
            rtt_media = []
            #jitter é a variação do rtt 

            for linha in linhas:
                latencia = self.encontrar_latencia(linha)
                if latencia is not None:
                    rtt_host.append(latencia)
                    print(f'tempo: {latencia}ms')

            if rtt_host:
                valores_rtt = [int(rtt) for rtt in rtt_host]
                media = sum(valores_rtt) / len(valores_rtt)
                return rtt_host

            else:
                return []
        else:
            print(f'pacote ECHO comprometido: {host} - ({result.returncode})')
            return []






    #def rtts(self,rtt):

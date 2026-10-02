import subprocess
import platform
import json
from json import JSONDecodeError
import time
from datetime import datetime
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from mpl_toolkits.axes_grid1 import host_subplot
from numpy.f2py.crackfortran import endifs

#tomar, controlar, superar e jamais recurar

system = platform.system()
tempo = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# ===================== caminhos ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "pops.json")
output_file = os.path.join(base_dir, "output.txt")
rtts_path = os.path.join(base_dir, "rtts.dat") #volatil
stats_path = os.path.join(base_dir, "ping_statistics.json") #permanente
debug_path = os.path.join(base_dir,'debug.txt') #para quaisquer saídas do programa

# ===========================================================

class RavenSniffer:
    def __int__(self,config_file, ping_interval: int = 2):
        self.config_file = data_path
        self.ping_interval = ping_interval
        self.hosts = []
        self.results = {}
        self.running = False
        self.rtt = []
        self.cache = []
        self.lock = threading.lock()
        self.debug = {}

        self.stats_file = stats_path #carrega o arquivo das estatisticas
        self.history_data = self.carregar_historico() #abre as estatisticas

        self.rtts_file = rtts_path



    def carregar_historico(self) -> Dict:
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r',encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                with open(debug_path, 'r', encoding='utf-8') as f:
                    f.write('=' * 10)
                    f.write('\n')
                    f.write(f'Iteração {tempo} falhou: {e}\n')

            except JSONDecodeError as e:
                with open(debug_path, 'r', encoding='utf-8') as f:
                    f.write('=' * 10)
                    f.write('\n')
                    f.write(f'Iteração {tempo} falhou: {e}\n')

        else:
            with open(debug_path, 'r', encoding='utf-8') as f: #caso o caminho da estatistica não for encontrado
                f.write('=' * 10)
                f.write('\n')
                f.write(f'Iteração {tempo} falhou: self.stats_file não foi encontrado.')



    def salvar_historico(self):
        with self.lock:
            with open(self.stats_file, 'w',encoding='utf-8') as f:
                json.dump(self.history_data, f, indent=2, ensure_ascii=False)




    def _save_rtt_volatile(self, session_data: Dict):
        with self.lock:
            with open(self.rtts_file, 'w',encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

    def carregar_rede(self):
        try:
            with open(data_path, 'r',encoding='utf-8') as f:
                rede = json.load(f)

            for noh in rede:
                for roteador in noh['routers']:
                    self.hosts.append({
                        'nome':roteador['name'],
                        'ip':roteador['ip']
                    })

        except FileNotFoundError as e:
            with open(debug_path, 'r', encoding='utf-8') as f:
                f.write('=' * 10)
                f.write('\n')
                f.write(f'Iteração {tempo} falhou: {e}\n')
            return False

        except JSONDecodeError as e:
            with open(debug_path, 'r', encoding='utf-8') as f:
                f.write('=' * 10)
                f.write('\n')
                f.write(f'Iteração {tempo} falhou: {e}\n')
            return False

        except KeyError as e:
            with open(debug_path, 'r', encoding='utf-8') as f:
                f.write('=' * 10)
                f.write('\n')
                f.write(f'Iteração {tempo} falhou: {e}\n')
            return False

        except Exception as e:
            with open(debug_path, 'r', encoding='utf-8') as f:
                f.write('=' * 10)
                f.write('\n')
                f.write(f'Iteração {tempo} falhou: {e}\n')
            return False

    def extrair_latencia(selfs,ping_output: str, host: str) -> Optional[float]:
        if system == "Windows":
            # Windows: "Média = 15ms" ou "Average = 15ms"
            patterns = [
                r'Média = (\d+)ms',
                r'Average = (\d+)ms',
                r'Média = (\d+\.?\d*)ms',
                r'Average = (\d+\.?\d*)ms'
            ]
            for pattern in patterns:
                match = re.search(pattern, ping_output, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        else:  # Linux
            # Linux: "rtt min/avg/max/mdev = 14.325/15.234/16.123/0.456 ms"
            pattern = r'rtt.*=\s+[\d\.]+/([\d\.]+)/[\d\.]+/[\d\.]+\s+ms'
            match = re.search(pattern, ping_output)
            if match:
                return float(match.group(1))

            # Fallback para tempos individuais
            pattern = r'time=(\d+\.?\d*)\s*ms'
            times = re.findall(pattern, ping_output)
            if times:
                return sum(float(t) for t in times) / len(times)

        return None

    def pinger(self,host_info):
        host = host_info['ip']
        nome = host_info['name']

        if system == "Windows":
            ping_cmd = ['ping', '-n', '4', '-w', '1000', host]
        else:
            ping_cmd = ['ping', '-c', '4', '-W', '1', host]

        try:
            inicio_tempo = time.time()
            result = subprocess.run(
                ping_cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            fim_tempo = time.time()
            tempo_resposta_ms = (fim_tempo - inicio_tempo) * 1000 #unidade em milissegundo

            sucesso = result.returncode == 0

            latencia = None
            if sucesso:
                latencia = self.extrair_latencia(result.stdout, host)

            return {
                'name':nome,
                'ip':host,
                'sucesso':sucesso,
                'latencia':latencia if latencia else (tempo_resposta_ms if sucesso else None),
                'output':result.stdout[:200] if sucesso else result.stderr[:200]
            }

        except subprocess.TimeoutExpired as e:
                with open(debug_path, 'r', encoding='utf-8') as f:
                    f.write('=' * 10)
                    f.write('\n')
                    f.write(f'Iteração {tempo} falhou: {e}\n')

                return {
                    'name':nome,
                    'ip':host,
                    'sucesso':False,
                    'latencia':None,
                    'tempo':datetime.now().isoformat(),
                    'erro':f'Timeout após 5 segundos.'
                }

        except Exception as e:
            with open(debug_path, 'r', encoding='utf-8') as f:
                f.write('=' * 10)
                f.write('\n')
                f.write(f'Iteração {tempo} falhou: {e}\n')

            return {
                'nome':nome,
                'ip':host,
                'sucesso':False,
                'latencia':None,
                'tempo':datetime.now().isoformat(),
                'erro':str(e)

            }

    def ping_thread(self, carga: int = 10):
        if not self.hosts:
            with open(debug_path, 'r', encoding='utf-8') as f:
                f.write('=' * 10)
                f.write('\n')
                f.write(f'Iteração {tempo}: nenhum host carregado.\n')
                f.write(f"\n▶ ping em {len(self.hosts)} hosts com {carga} threads...")

            results = []



        with ThreadPoolExecutor(max_workers=carga) as executor:
            future_to_host = {
                executor.submit(self.pinger, host)

            }







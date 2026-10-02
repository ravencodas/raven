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

# tomar, controlar, superar e jamais recuar


# tempo aqui: (3 + (0,38*60)) + (1 + (0,51*60))
system = platform.system()
tempo = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# ===================== caminhos ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "pops.json")
output_file = os.path.join(base_dir, "output.txt")
rtts_path = os.path.join(base_dir, "rtts.txt")
stats_path = os.path.join(base_dir, "ping_statistics.json")  # Arquivo permanente


# =============================================================================================
class RavenSniffer:
    def __init__(self, config_file, ping_interval: int = 5):
        self.config_file = data_path
        self.ping_interval = ping_interval
        self.hosts = []
        self.results = {}
        self.running = False
        self.rtt = []
        self.lock = threading.Lock()  # Para acesso thread-safe

        # Carrega histórico de estatísticas
        self.stats_file = stats_path
        self.history_data = self._load_history()

        # Arquivo volátil será sobrescrito a cada execução
        self.rtts_file = rtts_path

    def _load_history(self) -> Dict:
        """Carrega histórico de estatísticas do arquivo permanente"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (JSONDecodeError, Exception):
                return {'hosts': {}, 'sessions': []}
        else:
            return {'hosts': {}, 'sessions': []}

    def _save_history(self):
        """Salva histórico no arquivo permanente"""
        with self.lock:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.history_data, f, indent=2, ensure_ascii=False)

    def _save_rtt_volatile(self, session_data: Dict):
        """Salva dados no arquivo volátil (sobrescreve a cada execução)"""
        with self.lock:
            with open(self.rtts_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print(f"   Dados voláteis salvos em: {self.rtts_file}")

    def load_network(self):  # carrega os dados do json
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                network = json.load(f)

            for dot in network:  # a network é o json bruto, o dot é o item "routers"
                for router in dot["routers"]:
                    self.hosts.append({  # cada item da lista
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

        except KeyError as e:
            print(f"ERRO: estrutura do JSON inválida")
            print(f"   Chave ausente: {e}")
            return False

        except Exception as e:
            print(f"ERRO: {type(e).__name__} - {e}")
            return False

    def extract_latency(self, ping_output: str, host: str) -> Optional[float]:
        """Extrai latência média do output do ping"""
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

    def ping_host(self, host_info: Dict) -> Dict:
        """Pinga um único host e retorna com latência"""
        host = host_info['ip']
        name = host_info['name']

        # Define comando conforme sistema operacional
        if system == "Windows":
            ping_cmd = ['ping', '-n', '4', '-w', '1000', host]
        else:
            ping_cmd = ['ping', '-c', '4', '-W', '1', host]

        try:
            start_time = time.time()
            result = subprocess.run(
                ping_cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            success = result.returncode == 0

            # Extrai latência
            latency = None
            if success:
                latency = self.extract_latency(result.stdout, host)

            return {
                'name': name,
                'ip': host,
                'success': success,
                'latency_ms': latency if latency else (response_time_ms if success else None),
                'timestamp': datetime.now().isoformat(),
                'output': result.stdout[:200] if success else result.stderr[:200]
            }

        except subprocess.TimeoutExpired:
            return {
                'name': name,
                'ip': host,
                'success': False,
                'latency_ms': None,
                'timestamp': datetime.now().isoformat(),
                'error': 'Timeout após 5 segundos'
            }
        except Exception as e:
            return {
                'name': name,
                'ip': host,
                'success': False,
                'latency_ms': None,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def ping_all_hosts(self, max_workers: int = 10) -> List[Dict]:
        """Pinga todos os hosts usando threading"""
        if not self.hosts:
            print("Nenhum host carregado. Execute load_network() primeiro.")
            return []

        print(f"\n▶ Iniciando ping em {len(self.hosts)} hosts com {max_workers} threads...")
        print(f"   Sistema: {system}")
        print(f"   Timestamp: {tempo}\n")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_host = {
                executor.submit(self.ping_host, host): host
                for host in self.hosts
            }

            completed = 0
            for future in as_completed(future_to_host):
                result = future.result()
                results.append(result)
                completed += 1

                # Progresso formatado
                if result['success'] and result['latency_ms']:
                    status = f"✓ {result['latency_ms']:.1f}ms"
                elif result['success']:
                    status = "✓ (latência N/A)"
                else:
                    status = "✗ FALHA"

                print(f"[{completed:3}/{len(self.hosts)}] {result['name']:25} - {status}")

        return results

    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calcula estatísticas dos resultados"""
        successful = [r for r in results if r['success']]
        latencies = [r['latency_ms'] for r in successful if r['latency_ms'] is not None]

        statistics = {
            'timestamp': tempo,
            'datetime_iso': datetime.now().isoformat(),
            'total_hosts': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'success_rate': (len(successful) / len(results)) * 100 if results else 0,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else None,
            'min_latency_ms': min(latencies) if latencies else None,
            'max_latency_ms': max(latencies) if latencies else None,
            'results': results
        }

        return statistics

    def update_permanent_stats(self, session_stats: Dict):
        """Atualiza o arquivo permanente com estatísticas históricas"""
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        with self.lock:
            # Adiciona sessão ao histórico
            self.history_data.setdefault('sessions', []).append({
                'session_id': session_id,
                'timestamp': session_stats['datetime_iso'],
                'total_hosts': session_stats['total_hosts'],
                'successful': session_stats['successful'],
                'failed': session_stats['failed'],
                'avg_latency_ms': session_stats['avg_latency_ms']
            })

            # Atualiza estatísticas por host
            for result in session_stats['results']:
                host_key = result['ip']

                if host_key not in self.history_data['hosts']:
                    self.history_data['hosts'][host_key] = {
                        'name': result['name'],
                        'ip': result['ip'],
                        'first_seen': result['timestamp'],
                        'pings': [],
                        'success_count': 0,
                        'fail_count': 0,
                        'latencies': []
                    }

                host_stats = self.history_data['hosts'][host_key]
                host_stats['pings'].append({
                    'timestamp': result['timestamp'],
                    'success': result['success'],
                    'latency_ms': result['latency_ms']
                })

                if result['success']:
                    host_stats['success_count'] += 1
                    if result['latency_ms']:
                        host_stats['latencies'].append(result['latency_ms'])
                else:
                    host_stats['fail_count'] += 1

                # Calcula médias históricas
                if host_stats['latencies']:
                    host_stats['avg_latency_historical'] = sum(host_stats['latencies']) / len(host_stats['latencies'])
                    host_stats['min_latency_historical'] = min(host_stats['latencies'])
                    host_stats['max_latency_historical'] = max(host_stats['latencies'])

                total_pings = host_stats['success_count'] + host_stats['fail_count']
                host_stats['success_rate'] = (host_stats['success_count'] / total_pings) * 100 if total_pings > 0 else 0

            # Limita histórico para não crescer infinito (mantém últimas 100 sessões)
            if len(self.history_data['sessions']) > 100:
                self.history_data['sessions'] = self.history_data['sessions'][-100:]

            self._save_history()

    def run_ping_session(self, max_workers: int = 10) -> Dict:
        """Executa uma sessão completa de ping"""
        # Carrega rede se necessário
        if not self.hosts:
            if not self.load_network():
                return {}

        # Executa pings
        results = self.ping_all_hosts(max_workers=max_workers)

        # Calcula estatísticas da sessão
        session_stats = self.calculate_statistics(results)

        # Salva no arquivo volátil (sobrescreve)
        self._save_rtt_volatile(session_stats)

        # Atualiza estatísticas permanentes
        self.update_permanent_stats(session_stats)

        # Exibe resumo
        self.print_session_summary(session_stats)

        return session_stats

    def print_session_summary(self, session_stats: Dict):
        """Exibe resumo formatado da sessão"""
        print("\n" + "=" * 70)
        print("RESUMO DA SESSÃO DE PING")
        print("=" * 70)
        print(f"Timestamp: {session_stats['timestamp']}")
        print(f"Total hosts: {session_stats['total_hosts']}")
        print(f"✓ Sucessos: {session_stats['successful']}")
        print(f"✗ Falhas: {session_stats['failed']}")
        print(f"Taxa de sucesso: {session_stats['success_rate']:.1f}%")

        if session_stats['avg_latency_ms']:
            print(f"\n📊 ESTATÍSTICAS DE LATÊNCIA:")
            print(f"   Média: {session_stats['avg_latency_ms']:.1f}ms")
            print(f"   Mínima: {session_stats['min_latency_ms']:.1f}ms")
            print(f"   Máxima: {session_stats['max_latency_ms']:.1f}ms")

        print("\n📁 ARQUIVOS GERADOS:")
        print(f"   Volátil (esta sessão): {self.rtts_file}")
        print(f"   Permanente (histórico): {self.stats_file}")

    def get_host_history(self, ip: str = None) -> Dict:
        """Retorna histórico de um host específico ou de todos"""
        if ip:
            return self.history_data['hosts'].get(ip, {})
        return self.history_data['hosts']

    def print_host_report(self):
        """Exibe relatório histórico de todos os hosts"""
        if not self.history_data['hosts']:
            print("Nenhum dado histórico disponível.")
            return

        print("\n" + "=" * 90)
        print("RELATÓRIO HISTÓRICO DE LATÊNCIA")
        print("=" * 90)
        print(f"{'HOST':<30} {'SUCESSO':<10} {'FALHA':<10} {'TAXA (%)':<12} {'MÉDIA (ms)':<12}")
        print("-" * 90)

        for ip, stats in self.history_data['hosts'].items():
            name = stats['name'][:28]
            avg_lat = f"{stats.get('avg_latency_historical', 0):.1f}" if stats.get('avg_latency_historical') else "N/A"

            print(f"{name:<30} {stats['success_count']:<10} {stats['fail_count']:<10} "
                  f"{stats['success_rate']:<12.1f} {avg_lat:<12}")

    def cleanup_rtt(self):
        """Remove o arquivo volátil (opcional - pode ser chamado ao encerrar)"""
        if os.path.exists(self.rtts_file):
            try:
                os.remove(self.rtts_file)
                print(f"\nArquivo volátil removido: {self.rtts_file}")
            except Exception as e:
                print(f"Erro ao remover arquivo volátil: {e}")
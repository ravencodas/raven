
#usando o MAC como chave do dicionário

class MemoriaTmp:

    def __init__(self):
        self.dispositivos = {}

        self.interfaces = {}

        self.arp_obs = []

        self.session = {
            "iniciou em": None,
            "interface": None,
            "status": "inicializando..."

        }

    def add_dispositivo(self,dispositivo):

        self.dispositivos[dispositivo.mac] = dispositivo

    def ack_dispositivo(self,mac):

        return self.dispositivos.get(mac)

    def ack_all_dispositivos(self):

        return list(self.dispositivos.values())

    def clear(self):

        self.devices.clear()
        self.interfaces.clear()
        self.arp_obs.clear()
        self.session.clear()
        


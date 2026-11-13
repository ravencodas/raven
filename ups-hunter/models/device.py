#entidade para identificação de dispositivo durante a comuniação ethernet

from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Device:

    ip: str
    mac: str

    hostname: str | None = None
    vendor: str | None = None
    role: str = "desconhecido"

    visto_primeiro: datetime = field(
        default_factory=datetime.now
        )

    visto_ultimo: datetime = field(
        default_factory=datetime.now

    )

    interface: str | None = None

    
# limen/quantum_channel/teleport.py
# Compatibility shim — canonical implementation is in limen.communication.channel.
from limen.communication.channel import (  # noqa: F401
    TeleportResult,
    teleport_circuit,
    run_teleport_qpu,
)

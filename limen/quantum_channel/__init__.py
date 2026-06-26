# limen/quantum_channel/__init__.py
# Compatibility re-export layer.  All symbols live in limen.communication.
from limen.communication.channel import (  # noqa: F401
    ChannelDeltaModel,
    SiftedKeyResult,
    TeleportResult,
    bb84_circuit,
    estimate_fidelity,
    run_teleport_qpu,
    sift_and_evaluate,
    teleport_circuit,
)

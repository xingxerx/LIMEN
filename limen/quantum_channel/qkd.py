# limen/quantum_channel/qkd.py
# Compatibility shim — canonical implementation is in limen.communication.channel.
# QKDResult is exported here as an alias for SiftedKeyResult to preserve
# backward compatibility for code that imported it from this old path.
from limen.communication.channel import (  # noqa: F401
    SiftedKeyResult as QKDResult,
    bb84_circuit,
    sift_and_evaluate,
)

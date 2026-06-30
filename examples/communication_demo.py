# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Demo showcasing Quantum Communication in LIMEN.

Demonstrates state teleportation with verification and QKD (BB84)
key generation with/without eavesdropping.
"""

from __future__ import annotations

import math
from limen.communication.channel import QuantumChannel


def main() -> None:
    print("=" * 60)
    print("               LIMEN Quantum Communication Demo")
    print("=" * 60)

    try:
        import qiskit  # noqa: F401
    except ImportError:
        print("Error: The Qiskit SDK is required to run this demo.")
        print("Please install it using: pip install qiskit qiskit-aer")
        return

    # Initialize a local quantum channel using the noiseless statevector simulator
    print("\n[1/3] Initializing QuantumChannel...")
    channel = QuantumChannel(backend_name="statevector", seed=42)
    print("QuantumChannel active on backend: 'statevector'")

    # --- 1. Quantum Teleportation Demo ---
    print("\n[2/3] Executing Quantum Teleportation...")
    # Teleport a superposition state: |psi> = cos(theta/2)|0> + e^(i phi)sin(theta/2)|1>
    # e.g., theta = pi/3, phi = pi/4
    theta = math.pi / 3.0
    phi = math.pi / 4.0
    print(f"Preparing input state at Node A: theta = {theta:.4f} rad, phi = {phi:.4f} rad")

    # Run teleportation with inverse preparation verification enabled
    tel_res = channel.teleport(theta=theta, phi=phi, shots=1000, inverse_prep=True)

    print(f"-> Teleportation circuit depth: {tel_res.circuit_depth}")
    print(f"-> Verification Success Rate:  {tel_res.verification_success_rate * 100:.2f}%")
    print(f"-> Calculated State Fidelity:   {tel_res.fidelity * 100:.2f}%" if tel_res.fidelity is not None else "")

    # --- 2. QKD (BB84) Demo ---
    print("\n[3/3] Executing Quantum Key Distribution (BB84)...")

    # Scenario A: Secure channel (No eavesdropper)
    print("\n--- Scenario A: Clean Channel (No Eavesdropper) ---")
    qkd_res_clean = channel.qkd_bb84(key_length=80, eavesdrop_rate=0.0)
    print(f"-> Raw keys sent:          {qkd_res_clean.raw_key_length} bits")
    print(f"-> Sifted keys remaining:  {qkd_res_clean.sifted_key_length} bits")
    print(f"-> Estimated QBER:         {qkd_res_clean.qber * 100:.2f}%")
    print(f"-> Security status:        {'SECURE' if qkd_res_clean.secure else 'ABORTED'}")
    if qkd_res_clean.secure and qkd_res_clean.shared_key:
        print(f"-> Generated Key:          {qkd_res_clean.shared_key} (length: {len(qkd_res_clean.shared_key)})")

    # Scenario B: Compromised channel (Eve eavesdropping 100% of transmissions)
    print("\n--- Scenario B: Compromised Channel (Eve Eavesdropping) ---")
    qkd_res_compromised = channel.qkd_bb84(key_length=80, eavesdrop_rate=1.0)
    print(f"-> Raw keys sent:          {qkd_res_compromised.raw_key_length} bits")
    print(f"-> Sifted keys remaining:  {qkd_res_compromised.sifted_key_length} bits")
    print(f"-> Estimated QBER:         {qkd_res_compromised.qber * 100:.2f}%")
    print(f"-> Security status:        {'SECURE' if qkd_res_compromised.secure else 'ABORTED (Eavesdropper Detected!)'}")
    if qkd_res_compromised.shared_key:
        print(f"-> Generated Key:          {qkd_res_compromised.shared_key}")
    else:
        print("-> Generated Key:          None (Aborted for safety)")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

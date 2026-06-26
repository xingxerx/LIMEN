# Copyright 2026 LIMEN Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantum communication channel primitives for LIMEN.

Implements quantum teleportation and Quantum Key Distribution (QKD) BB84 protocols
run on simulators or physical QPU backends using Qiskit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

_INSTALL_MSG = (
    "The Qiskit SDK is required to use the QuantumChannel. "
    "Install it with: pip install limen[ibm]  "
    "(or: pip install qiskit qiskit-aer)"
)


def _check_qiskit() -> None:
    """Raise ImportError if qiskit is not installed."""
    try:
        import qiskit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


@dataclass
class TeleportationResult:
    """The result of a quantum teleportation execution.

    Attributes:
        input_state: Parameters of the input state prepared at Node A (theta, phi).
        circuit_depth: Transpiled circuit depth when available, else None.
        fidelity: Exact simulated fidelity (available only in exact simulation).
        verification_success_rate: Fraction of successful runs under inverse-prep validation.
        measured_counts: Raw measurement counts for Bob's qubit.
        metadata: Backend options, seeds, and execution telemetry.
    """

    input_state: dict[str, float]
    circuit_depth: int | None
    fidelity: float | None
    verification_success_rate: float | None
    measured_counts: dict[str, int] | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QKDResult:
    """The result of a Quantum Key Distribution run.

    Attributes:
        protocol: The protocol executed (e.g. "bb84").
        raw_key_length: Initial length of key bits transmitted.
        sifted_key_length: Length of sifted key bits remaining after basis comparison.
        qber: Estimated Quantum Bit Error Rate.
        secure: True if QBER is below the abort threshold (11%).
        shared_key: The final secret key as a binary string, or None if aborted/empty.
        metadata: Bases, test indices, and simulation flags.
    """

    protocol: Literal["bb84"]
    raw_key_length: int
    sifted_key_length: int
    qber: float
    secure: bool
    shared_key: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class QuantumChannel:
    """Emulates or executes quantum communication protocols on Qiskit backends."""

    def __init__(
        self,
        backend_name: str = "aer_simulator",
        use_runtime: bool = False,
        ibm_token: str | None = None,
        ibm_instance: str | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize the QuantumChannel connection.

        Args:
            backend_name: Target Qiskit backend or "statevector" for ideal simulation.
            use_runtime: Submit to real IBM QPU via QiskitRuntimeService.
            ibm_token: IBM Quantum Platform API token.
            ibm_instance: IBM Quantum CRN instance string.
            seed: RNG seed for deterministic runs.
        """
        self.backend_name = backend_name
        self.use_runtime = use_runtime
        self.ibm_token = ibm_token
        self.ibm_instance = ibm_instance
        self.seed = seed

    def _get_backend(self) -> Any:
        """Fetch the Qiskit backend client."""
        _check_qiskit()
        if self.use_runtime:
            from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=self.ibm_token,
                instance=self.ibm_instance,
            )
            return service.backend(self.backend_name)
        elif self.backend_name == "statevector":
            from qiskit_aer import AerSimulator  # type: ignore[import]
            return AerSimulator(method="statevector", seed_simulator=self.seed)
        else:
            from qiskit_aer import AerSimulator  # type: ignore[import]
            return AerSimulator(seed_simulator=self.seed)

    def teleport(
        self,
        theta: float,
        phi: float,
        shots: int = 1000,
        inverse_prep: bool = True,
    ) -> TeleportationResult:
        """Teleport an arbitrary state cos(theta/2)|0> + e^(i phi)sin(theta/2)|1>.

        Args:
            theta: Polar angle theta of the target state.
            phi: Azimuthal angle phi of the target state.
            shots: Measurement shots.
            inverse_prep: Apply the inverse of the preparation gate at Bob's qubit.
                If True, measuring 0 on Bob's qubit indicates a successful teleportation,
                serving as a direct physical validation metric.

        Returns:
            TeleportationResult.
        """
        _check_qiskit()
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile  # type: ignore[import]
        from qiskit.circuit.library import UGate  # type: ignore[import]

        # 3 qubits: q0 (Alice input), q1 (Alice entanglement), q2 (Bob entanglement)
        qr = QuantumRegister(3, name="q")
        cr0 = ClassicalRegister(1, name="cr0")
        cr1 = ClassicalRegister(1, name="cr1")
        cr2 = ClassicalRegister(1, name="cr2")
        qc = QuantumCircuit(qr, cr0, cr1, cr2)

        # 1. State preparation on q0
        qc.u(theta, phi, 0.0, qr[0])

        # 2. Shared EPR pair on q1 and q2
        qc.h(qr[1])
        qc.cx(qr[1], qr[2])

        # 3. Bell basis measurement at Alice (Node A)
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.measure(qr[0], cr0[0])
        qc.measure(qr[1], cr1[0])

        # 4. Classical feedforward correction to Bob (Node B)
        with qc.if_test((cr1, 1)):
            qc.x(qr[2])
        with qc.if_test((cr0, 1)):
            qc.z(qr[2])

        # 5. Optional verification: apply inverse state preparation and measure
        if inverse_prep:
            u_gate = UGate(theta, phi, 0.0)
            u_gate_inv = u_gate.inverse()
            qc.append(u_gate_inv, [qr[2]])

        qc.measure(qr[2], cr2[0])

        backend = self._get_backend()
        transpiled_qc = transpile(qc, backend)
        depth = transpiled_qc.depth()

        if self.use_runtime:
            from qiskit_ibm_runtime import SamplerV2  # type: ignore[import]
            sampler = SamplerV2(mode=backend)
            job = sampler.run([transpiled_qc], shots=shots)
            job_result = job.result()
            counts = job_result[0].data.cr2.get_counts()
        else:
            job = backend.run(transpiled_qc, shots=shots)
            counts = job.result().get_counts()

        # Parse counts to compute verification success rate
        # Keys of counts are space-separated bitstrings of register values: "cr2 cr1 cr0"
        # Since cr2 was added last, it is the leftmost value (index 0 when clean of spaces)
        success_count = 0
        total_count = 0
        for key, count in counts.items():
            clean_key = key.replace(" ", "")
            if clean_key[0] == "0":
                success_count += count
            total_count += count

        success_rate = success_count / total_count if total_count > 0 else 0.0

        # Exact noiseless simulation gets a theoretical fidelity from success_rate
        fidelity = success_rate if inverse_prep else None

        return TeleportationResult(
            input_state={"theta": theta, "phi": phi},
            circuit_depth=depth,
            fidelity=fidelity,
            verification_success_rate=success_rate if inverse_prep else None,
            measured_counts=counts,
            metadata={
                "backend": self.backend_name,
                "shots": shots,
                "inverse_prep": inverse_prep,
                "seed": self.seed,
            },
        )

    def qkd_bb84(
        self,
        key_length: int = 100,
        eavesdrop_rate: float = 0.0,
        shots: int = 1,
    ) -> QKDResult:
        """Run the BB84 protocol over the quantum channel.

        Args:
            key_length: Number of candidate key bits to transmit.
            eavesdrop_rate: Probability (0.0 to 1.0) of Eve intercepting and measuring.
            shots: Shots per circuit (default 1).

        Returns:
            QKDResult.
        """
        _check_qiskit()
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile  # type: ignore[import]

        rng = random.Random(self.seed)

        alice_bits = [rng.randint(0, 1) for _ in range(key_length)]
        alice_bases = [rng.choice(["Z", "X"]) for _ in range(key_length)]
        bob_bases = [rng.choice(["Z", "X"]) for _ in range(key_length)]

        circuits = []
        for i in range(key_length):
            qr = QuantumRegister(1, name="q")
            bob_cr = ClassicalRegister(1, name="bob")

            if eavesdrop_rate > 0.0:
                eve_cr = ClassicalRegister(1, name="eve")
                qc = QuantumCircuit(qr, bob_cr, eve_cr)
            else:
                qc = QuantumCircuit(qr, bob_cr)

            # Alice state preparation
            if alice_bits[i] == 1:
                qc.x(qr[0])
            if alice_bases[i] == "X":
                qc.h(qr[0])

            # Active eavesdropping by Eve
            if eavesdrop_rate > 0.0 and rng.random() < eavesdrop_rate:
                eve_basis = rng.choice(["Z", "X"])
                if eve_basis == "X":
                    qc.h(qr[0])
                    qc.measure(qr[0], eve_cr[0])
                    qc.h(qr[0])
                else:
                    qc.measure(qr[0], eve_cr[0])

            # Bob measurement
            if bob_bases[i] == "X":
                qc.h(qr[0])
            qc.measure(qr[0], bob_cr[0])

            circuits.append(qc)

        backend = self._get_backend()
        transpiled_circuits = transpile(circuits, backend)

        bob_results = []
        if self.use_runtime:
            from qiskit_ibm_runtime import SamplerV2  # type: ignore[import]
            sampler = SamplerV2(mode=backend)
            job = sampler.run(transpiled_circuits, shots=shots)
            job_result = job.result()
            for j in range(key_length):
                counts = job_result[j].data.bob.get_counts()
                bit = max(counts, key=counts.get)
                bob_results.append(int(bit))
        else:
            job = backend.run(transpiled_circuits, shots=shots)
            job_result = job.result()
            for j in range(key_length):
                counts = job_result.get_counts(j)
                bit_str = max(counts, key=counts.get)
                clean_bit_str = bit_str.replace(" ", "")
                # Since bob_cr is index 0, it is rightmost in Qiskit bitstring order
                bob_results.append(int(clean_bit_str[-1]))

        # Sifting
        sifted_indices = [
            i for i in range(key_length) if alice_bases[i] == bob_bases[i]
        ]
        sifted_key_length = len(sifted_indices)

        if sifted_key_length == 0:
            return QKDResult(
                protocol="bb84",
                raw_key_length=key_length,
                sifted_key_length=0,
                qber=0.0,
                secure=False,
                shared_key=None,
                metadata={"error": "No bases matched during sifting."},
            )

        # Estimate Quantum Bit Error Rate (QBER) on half the sifted key.
        # A 10% sample is too small to reliably distinguish a noiseless
        # channel from one with ~25% disturbance (e.g. full eavesdropping
        # only disturbs the basis-matched subset half the time).
        num_test_bits = max(1, sifted_key_length // 2)
        test_indices = rng.sample(sifted_indices, num_test_bits)

        errors = 0
        for idx in test_indices:
            if alice_bits[idx] != bob_results[idx]:
                errors += 1
        qber = errors / num_test_bits

        # Standard secure QBER threshold is 11%
        secure = qber <= 0.11

        final_key_indices = [idx for idx in sifted_indices if idx not in test_indices]
        if secure and final_key_indices:
            final_key = "".join(str(alice_bits[idx]) for idx in final_key_indices)
        else:
            final_key = None

        return QKDResult(
            protocol="bb84",
            raw_key_length=key_length,
            sifted_key_length=sifted_key_length,
            qber=qber,
            secure=secure,
            shared_key=final_key,
            metadata={
                "alice_bases": alice_bases,
                "bob_bases": bob_bases,
                "test_indices": test_indices,
                "eavesdrop_rate": eavesdrop_rate,
                "seed": self.seed,
            },
        )

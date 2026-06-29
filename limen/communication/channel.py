# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Quantum communication channel primitives for LIMEN.

Implements quantum teleportation and Quantum Key Distribution (QKD) BB84 protocols
run on simulators or physical QPU backends using Qiskit.

This is the single canonical module for all quantum-channel functionality.
``limen.quantum_channel`` is a thin compatibility re-export layer that points here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

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


# ---------------------------------------------------------------------------
# Hardware-level channel model (feedforward latency / coherence)
# ---------------------------------------------------------------------------

@dataclass
class ChannelDeltaModel:
    """Models classical feedforward latency and its effect on qubit coherence.

    Analogous to HardwareDeltaModel but for the inter-node classical channel
    rather than QPU calibration.

    Attributes:
        latency_ms: Classical channel round-trip latency in milliseconds.
        t2_us: QPU T2 coherence time in microseconds.
        gate_time_us: Single-qubit gate time in microseconds.
    """

    latency_ms: float
    t2_us: float
    gate_time_us: float = 0.1

    def within_coherence(self) -> bool:
        """True if feedforward completes before T2 decay."""
        return (self.latency_ms * 1000.0) < self.t2_us

    def fidelity_penalty(self) -> float:
        """Exponential decay estimate: exp(-latency / T2)."""
        t_us = self.latency_ms * 1000.0
        return math.exp(-t_us / self.t2_us)

    def to_dict(self) -> dict:
        return {
            "latency_ms": self.latency_ms,
            "t2_us": self.t2_us,
            "gate_time_us": self.gate_time_us,
            "within_coherence": self.within_coherence(),
            "fidelity_penalty": self.fidelity_penalty(),
        }


# ---------------------------------------------------------------------------
# Low-level teleportation result (QPU execution path)
# ---------------------------------------------------------------------------

@dataclass
class TeleportResult:
    """Low-level result from a QPU teleportation job.

    Returned by :func:`run_teleport_qpu` and :func:`estimate_fidelity`.
    The high-level simulator path returns :class:`TeleportationResult` instead.

    Attributes:
        fidelity_estimate: Fraction of shots that matched the expected outcome.
        success: True if fidelity_estimate > 0.5.
        backend: Backend name used for the job.
        job_id: IBM job ID, if available.
        channel_delta: Optional coherence model derived from live T2 calibration.
    """

    fidelity_estimate: float
    success: bool
    backend: str
    job_id: Optional[str] = None
    channel_delta: Optional[ChannelDeltaModel] = None

    def to_dict(self) -> dict:
        return {
            "fidelity_estimate": self.fidelity_estimate,
            "success": self.success,
            "backend": self.backend,
            "job_id": self.job_id,
            "channel_delta": self.channel_delta.to_dict() if self.channel_delta else None,
        }


# ---------------------------------------------------------------------------
# Low-level sifted-key QKD result (basis-sifting path)
# ---------------------------------------------------------------------------

@dataclass
class SiftedKeyResult:
    """Result of the BB84 basis-sifting step returned by :func:`sift_and_evaluate`.

    This is the *low-level* result that carries raw key lists and per-index
    information.  The high-level :class:`QKDResult` returned by
    :meth:`QuantumChannel.qkd_bb84` carries the final shared key string instead.

    Attributes:
        raw_key: Alice's full bit string before sifting.
        sifted_key: Bits retained after basis matching.
        qber: Quantum Bit Error Rate over the sifted key.
        secure: True if QBER is below the 11 % abort threshold.
        backend: Backend identifier used for the job.
        job_id: Job ID, if submitted to a QPU.
    """

    raw_key: list[int]
    sifted_key: list[int]
    qber: float
    secure: bool
    backend: str
    job_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sifted_key_length": len(self.sifted_key),
            "qber": self.qber,
            "secure": self.secure,
            "backend": self.backend,
            "job_id": self.job_id,
        }


# ---------------------------------------------------------------------------
# Low-level circuit builders (gated on qiskit)
# ---------------------------------------------------------------------------

def teleport_circuit():
    """Build a standard 3-qubit teleportation circuit.

    q0 = state to send; q1/q2 = Bell pair (Node A / Node B).
    Requires qiskit; raises :exc:`ImportError` if not installed.
    """
    _check_qiskit()
    from qiskit import QuantumCircuit  # type: ignore[import]

    qc = QuantumCircuit(3, 3)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.measure(2, 2)
    return qc


def bb84_circuit(n_bits: int):
    """Build a BB84 QKD circuit for *n_bits*.

    Returns ``(circuit, alice_bases, alice_bits, bob_bases)``.
    Requires qiskit; raises :exc:`ImportError` if not installed.
    """
    _check_qiskit()
    from qiskit import QuantumCircuit  # type: ignore[import]

    alice_bits = [random.randint(0, 1) for _ in range(n_bits)]
    alice_bases = [random.randint(0, 1) for _ in range(n_bits)]
    bob_bases = [random.randint(0, 1) for _ in range(n_bits)]

    qc = QuantumCircuit(n_bits, n_bits)
    for i in range(n_bits):
        if alice_bits[i] == 1:
            qc.x(i)
        if alice_bases[i] == 1:
            qc.h(i)
    for i in range(n_bits):
        if bob_bases[i] == 1:
            qc.h(i)
    qc.measure(range(n_bits), range(n_bits))

    return qc, alice_bases, alice_bits, bob_bases


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def estimate_fidelity(
    counts: dict[str, int],
    backend: str,
    job_id: Optional[str] = None,
    expected_bit: str = "0",
) -> TeleportResult:
    """Estimate teleportation fidelity from raw IBM job measurement counts.

    :func:`teleport_circuit` always prepares q0 in |0>, so a faithful
    teleportation measures Bob's qubit (q2, classical bit 2) as
    ``expected_bit``.  Qiskit formats counts keys as ``"c2c1c0"``
    (bit 2 leftmost), so *fidelity_estimate* is the fraction of shots
    whose leftmost character matches *expected_bit*.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return TeleportResult(
            fidelity_estimate=0.0, success=False, backend=backend, job_id=job_id
        )

    correct_outcomes = sum(
        count
        for bitstring, count in counts.items()
        if bitstring.replace(" ", "")[0] == expected_bit
    )
    fidelity_estimate = correct_outcomes / total_shots
    return TeleportResult(
        fidelity_estimate=fidelity_estimate,
        success=fidelity_estimate > 0.5,
        backend=backend,
        job_id=job_id,
    )


def sift_and_evaluate(
    alice_bits: list[int],
    alice_bases: list[int],
    bob_bases: list[int],
    bob_results: list[int],
    backend: str = "classical",
    job_id: Optional[str] = None,
) -> SiftedKeyResult:
    """Sift keys on matching bases, compute QBER, return :class:`SiftedKeyResult`."""
    sifted_alice: list[int] = []
    sifted_bob: list[int] = []
    for i in range(len(alice_bases)):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_bits[i])
            sifted_bob.append(bob_results[i])

    if not sifted_alice:
        return SiftedKeyResult([], [], 1.0, False, backend, job_id)

    errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
    qber = errors / len(sifted_alice)

    return SiftedKeyResult(
        raw_key=alice_bits,
        sifted_key=sifted_alice,
        qber=qber,
        secure=qber < 0.11,
        backend=backend,
        job_id=job_id,
    )


# ---------------------------------------------------------------------------
# Pure-Python feedforward teleportation (no Qiskit required)
# ---------------------------------------------------------------------------

@dataclass
class FeedforwardTransport:
    """Classical transport of Bell-measurement bits from Alice to Bob.

    Models the classical communication step of the teleportation protocol:
    Alice measures her two qubits (obtaining bits m0 and m1), sends them to
    Bob over the classical channel, and Bob applies the conditional Pauli
    corrections.  The :class:`ChannelDeltaModel` governs whether this
    transport completes within the qubit's T2 coherence time.

    Attributes:
        alice_bits: ``(m0, m1)`` — Alice's measurement outcomes for q0 and q1.
        correction: Pauli correction applied to Bob's qubit:
            ``"I"``, ``"X"``, ``"Z"``, or ``"XZ"``.
        within_coherence: ``True`` if the transport latency is below T2;
            ``None`` if no :class:`ChannelDeltaModel` was provided.
        fidelity_penalty: ``exp(-latency / T2)`` decay factor;
            ``None`` if no :class:`ChannelDeltaModel` was provided.
        transport_latency_ms: Modelled classical channel latency in ms;
            ``None`` if no :class:`ChannelDeltaModel` was provided.
    """

    alice_bits: tuple[int, int]
    correction: str
    within_coherence: bool | None
    fidelity_penalty: float | None
    transport_latency_ms: float | None

    def to_dict(self) -> dict:
        return {
            "alice_bits": list(self.alice_bits),
            "correction": self.correction,
            "within_coherence": self.within_coherence,
            "fidelity_penalty": self.fidelity_penalty,
            "transport_latency_ms": self.transport_latency_ms,
        }


def correction_for_bits(m0: int, m1: int) -> str:
    """Return the Pauli correction string Bob applies for Alice's bits (m0, m1).

    Shared by the single-process path (:func:`simulate_feedforward_teleport`)
    and the cross-node RPC path (``CoordinationServicer.TransportFeedforward``
    in ``limen.distributed.server``) so both compute the same correction for
    the same measurement outcomes:

    - m1 == 1  ->  "X"
    - m0 == 1  ->  "Z"
    - both     ->  "XZ"
    - neither  ->  "I"
    """
    parts: list[str] = []
    if m1 == 1:
        parts.append("X")
    if m0 == 1:
        parts.append("Z")
    return "".join(parts) or "I"


def apply_bob_correction(
    state: list[complex], n: int, qubit: int, m0: int, m1: int
) -> str:
    """Apply Bob's Pauli correction for (m0, m1) to *qubit* of *state*, in place.

    Returns the correction string applied (see :func:`correction_for_bits`).
    """
    from limen.gates.simulator import apply_gate_to_state

    if m1 == 1:
        apply_gate_to_state(state, n, "x", [qubit], [])
    if m0 == 1:
        apply_gate_to_state(state, n, "z", [qubit], [])
    return correction_for_bits(m0, m1)


def simulate_feedforward_teleport(
    theta: float,
    phi: float,
    channel_delta: "ChannelDeltaModel | None" = None,
    seed: int = 42,
) -> "tuple[TeleportationResult, FeedforwardTransport]":
    """Teleport a qubit state using the pure-Python :class:`StatevectorSimulator`.

    Executes the full teleportation protocol with *explicit* classical
    feedforward transport — no Qiskit required:

    1. Prepare |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩ on qubit 0.
    2. Create a Bell pair on qubits 1 and 2.
    3. Apply the Bell-basis transformation on Alice's side (qubits 0–1).
    4. Measure q0 → m0, q1 → m1  (projective collapse of the statevector).
    5. Model classical transport of (m0, m1) via *channel_delta* if given.
    6. Apply Pauli corrections on Bob's qubit (q2):
       - m1 == 1  →  X on q2
       - m0 == 1  →  Z on q2
    7. Verify: apply the inverse state preparation on q2 and compute
       P(q2 = |0⟩), which equals the teleportation fidelity.

    The inverse of U(θ, φ, 0) is U(-θ, 0, -φ) under the ZYZ decomposition.

    Args:
        theta: Polar angle of the input state on the Bloch sphere.
        phi: Azimuthal angle of the input state.
        channel_delta: Optional coherence model.  If supplied, the returned
            :class:`FeedforwardTransport` includes ``within_coherence`` and
            ``fidelity_penalty`` values derived from the T2 and latency.
        seed: RNG seed for the projective measurement step (deterministic).

    Returns:
        ``(TeleportationResult, FeedforwardTransport)``
    """
    from limen.gates.ir import CircuitIR, GateInstruction
    from limen.gates.simulator import apply_gate_to_state, measure_qubit, statevector

    n = 3  # q0 = Alice's input state, q1 = Alice's EPR half, q2 = Bob's EPR half

    # Steps 1–3: state preparation + Bell pair + Bell-basis rotation
    pre = CircuitIR(n_qubits=n)
    pre.instructions = [
        GateInstruction("u", [0], [theta, phi, 0.0]),   # state prep on q0
        GateInstruction("h", [1], []),                   # Bell pair: H on q1
        GateInstruction("cx", [1, 2], []),               # Bell pair: CX q1→q2
        GateInstruction("cx", [0, 1], []),               # Bell-basis CNOT
        GateInstruction("h", [0], []),                   # Bell-basis Hadamard
    ]
    state = statevector(pre)

    # Step 4: projective measurement of Alice's qubits
    m0, state = measure_qubit(state, n, qubit=0, seed=seed)
    m1, state = measure_qubit(state, n, qubit=1, seed=seed ^ 0xFF)

    # Step 5: model classical feedforward transport
    within_coherence: bool | None = None
    fidelity_penalty: float | None = None
    transport_latency_ms: float | None = None
    if channel_delta is not None:
        within_coherence = channel_delta.within_coherence()
        fidelity_penalty = channel_delta.fidelity_penalty()
        transport_latency_ms = channel_delta.latency_ms

    # Step 6: Pauli corrections on Bob's qubit (q2) based on Alice's bits
    correction = apply_bob_correction(state, n, qubit=2, m0=m0, m1=m1)

    # Step 7: verify — inverse prep on q2, then measure P(q2 = |0⟩)
    # U(θ, φ, 0)† = U(-θ, 0, -φ)  [standard ZYZ inverse]
    apply_gate_to_state(state, n, "u", [2], [-theta, 0.0, -phi])

    prob_q2_zero = sum(
        state[k].real ** 2 + state[k].imag ** 2
        for k in range(1 << n)
        if not ((k >> 2) & 1)  # qubit 2 in |0⟩
    )
    fidelity = float(max(0.0, min(1.0, prob_q2_zero)))

    transport = FeedforwardTransport(
        alice_bits=(m0, m1),
        correction=correction,
        within_coherence=within_coherence,
        fidelity_penalty=fidelity_penalty,
        transport_latency_ms=transport_latency_ms,
    )
    result = TeleportationResult(
        input_state={"theta": theta, "phi": phi},
        circuit_depth=None,
        fidelity=fidelity,
        verification_success_rate=fidelity,
        measured_counts=None,
        metadata={
            "backend": "statevector_simulator",
            "seed": seed,
            "inverse_prep": True,
            "alice_bits": [m0, m1],
            "correction": correction,
        },
    )
    return result, transport


def run_distributed_feedforward_teleport(
    theta: float,
    phi: float,
    peer_address: str,
    seed: int = 42,
    t2_us: float = 100.0,
) -> "tuple[TeleportationResult, FeedforwardTransport]":
    """Teleport a state with Alice's classical bits transported to a peer node.

    This is the cross-process counterpart to :func:`simulate_feedforward_teleport`:
    Alice's side (state prep, Bell pair, Bell-basis measurement) runs locally
    on the pure-Python statevector simulator exactly as in steps 1–4 of
    :func:`simulate_feedforward_teleport`. Alice's measurement bits (m0, m1)
    are then sent to the peer LIMEN node at *peer_address* over the
    ``Coordination.TransportFeedforward`` gRPC RPC (see
    ``limen.distributed.client.CoordinationClient.transport_feedforward``).
    Round-trip wall-clock latency is measured and used to build the
    :class:`ChannelDeltaModel` returned in the :class:`FeedforwardTransport`,
    in place of the modelled constant latency used by the local-only path.

    The peer independently computes the correction for (m0, m1) (via
    :func:`correction_for_bits`) and echoes it back; this process then
    applies that same correction to its local copy of Bob's qubit to finish
    the fidelity verification, so the result is numerically equivalent to
    :func:`simulate_feedforward_teleport` while exercising real network I/O
    for the classical feedforward step.

    Args:
        theta: Polar angle of the input state.
        phi: Azimuthal angle of the input state.
        peer_address: ``"host:port"`` of the peer LIMEN node's Coordination
            service (Bob).
        seed: RNG seed for Alice's projective measurements (deterministic).
        t2_us: T2 coherence time (microseconds) used to build the
            :class:`ChannelDeltaModel` from the measured latency.

    Returns:
        ``(TeleportationResult, FeedforwardTransport)`` where
        ``FeedforwardTransport.transport_latency_ms`` is the *measured*
        round-trip latency of the ``TransportFeedforward`` RPC call.
    """
    from limen.distributed.client import CoordinationClient
    from limen.gates.ir import CircuitIR, GateInstruction
    from limen.gates.simulator import measure_qubit, statevector

    n = 3  # q0 = Alice's input state, q1 = Alice's EPR half, q2 = Bob's EPR half

    # Steps 1-3: state preparation + Bell pair + Bell-basis rotation (Alice).
    pre = CircuitIR(n_qubits=n)
    pre.instructions = [
        GateInstruction("u", [0], [theta, phi, 0.0]),
        GateInstruction("h", [1], []),
        GateInstruction("cx", [1, 2], []),
        GateInstruction("cx", [0, 1], []),
        GateInstruction("h", [0], []),
    ]
    state = statevector(pre)

    # Step 4: Alice's projective measurement.
    m0, state = measure_qubit(state, n, qubit=0, seed=seed)
    m1, state = measure_qubit(state, n, qubit=1, seed=seed ^ 0xFF)

    # Step 5: transport (m0, m1) to the peer node over the real network.
    client = CoordinationClient(peer_address)
    try:
        correction, channel_delta = client.transport_feedforward(
            m0, m1, theta=theta, phi=phi, t2_us=t2_us
        )
    finally:
        client.close()

    # Step 6: apply the peer-confirmed correction to Bob's qubit (q2) locally.
    from limen.gates.simulator import apply_gate_to_state

    if "X" in correction:
        apply_gate_to_state(state, n, "x", [2], [])
    if "Z" in correction:
        apply_gate_to_state(state, n, "z", [2], [])

    # Step 7: verify — inverse prep on q2, then measure P(q2 = |0>).
    apply_gate_to_state(state, n, "u", [2], [-theta, 0.0, -phi])

    prob_q2_zero = sum(
        state[k].real ** 2 + state[k].imag ** 2
        for k in range(1 << n)
        if not ((k >> 2) & 1)
    )
    fidelity = float(max(0.0, min(1.0, prob_q2_zero)))

    transport = FeedforwardTransport(
        alice_bits=(m0, m1),
        correction=correction,
        within_coherence=channel_delta.within_coherence() if channel_delta else None,
        fidelity_penalty=channel_delta.fidelity_penalty() if channel_delta else None,
        transport_latency_ms=channel_delta.latency_ms if channel_delta else None,
    )
    result = TeleportationResult(
        input_state={"theta": theta, "phi": phi},
        circuit_depth=None,
        fidelity=fidelity,
        verification_success_rate=fidelity,
        measured_counts=None,
        metadata={
            "backend": "statevector_simulator+grpc",
            "seed": seed,
            "inverse_prep": True,
            "alice_bits": [m0, m1],
            "correction": correction,
            "peer_address": peer_address,
        },
    )
    return result, transport


# ---------------------------------------------------------------------------
# Full QPU execution path
# ---------------------------------------------------------------------------

_DEFAULT_FEEDFORWARD_LATENCY_MS = 0.001


def run_teleport_qpu(
    token: str,
    crn: str,
    backend_name: str = "ibm_kingston",
    shots: int = 1000,
) -> TeleportResult:
    """Submit the teleportation circuit to a real IBM QPU and estimate fidelity.

    Requires qiskit and qiskit-ibm-runtime; raises :exc:`ImportError` if not
    installed.
    """
    _check_qiskit()
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # type: ignore[import]
    from qiskit.transpiler.preset_passmanagers import (  # type: ignore[import]
        generate_preset_pass_manager,
    )

    qc = teleport_circuit()

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=crn,
    )
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(qc)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    job_id = job.job_id()
    pub_result = job.result()[0]
    counts: dict[str, int] = pub_result.data.c.get_counts()

    result = estimate_fidelity(counts, backend=backend_name, job_id=job_id)
    result.channel_delta = _channel_delta_from_backend(backend)
    return result


def _channel_delta_from_backend(backend: Any) -> Optional[ChannelDeltaModel]:
    """Build a :class:`ChannelDeltaModel` from a backend's live T2 calibration."""
    t2_values = [
        props.t2
        for q in range(backend.num_qubits)
        if (props := backend.qubit_properties(q)) is not None and props.t2
    ]
    if not t2_values:
        return None
    median_t2_us = sorted(t2_values)[len(t2_values) // 2] * 1e6
    return ChannelDeltaModel(
        latency_ms=_DEFAULT_FEEDFORWARD_LATENCY_MS, t2_us=median_t2_us
    )

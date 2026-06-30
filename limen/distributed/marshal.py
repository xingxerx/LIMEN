# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Conversions between LIMEN dataclasses and their protobuf wire messages.

Built directly on the existing to_dict()/from_dict() convention (see
HardwareDeltaModel in limen/analog/delta_model.py) so the proto messages
mirror the JSON-safe dict shape rather than introducing a parallel model.

Note: metadata fields are constrained to string values on the wire
(map<string, string>); non-string metadata is coerced via str() and will
not round-trip to its original type. This is a deliberate scope limit for
milestone 1 - arbitrary metadata was not required by any current caller.
"""

from __future__ import annotations

from limen.analog.delta_model import DeviceDrift, HardwareDeltaModel
from limen.distributed.node import NodeInfo
from limen.distributed.proto import coordination_pb2 as pb


def node_info_to_proto(info: NodeInfo) -> pb.NodeInfo:
    return pb.NodeInfo(
        node_id=info.node_id,
        host=info.host,
        port=info.port,
        device_ids=list(info.device_ids),
    )


def node_info_from_proto(msg: pb.NodeInfo) -> NodeInfo:
    return NodeInfo(
        node_id=msg.node_id,
        host=msg.host,
        port=msg.port,
        device_ids=list(msg.device_ids),
    )


def drift_to_proto(drift: DeviceDrift) -> pb.DeviceDriftProto:
    d = drift.to_dict()
    return pb.DeviceDriftProto(
        site_detuning_offsets=d["site_detuning_offsets"],
        coupling_scale_errors=d["coupling_scale_errors"],
        global_rabi_error=d["global_rabi_error"],
        timestamp=d["timestamp"],
        metadata={k: str(v) for k, v in d["metadata"].items()},
    )


def drift_from_proto(msg: pb.DeviceDriftProto) -> DeviceDrift:
    return DeviceDrift.from_dict(
        {
            "site_detuning_offsets": dict(msg.site_detuning_offsets),
            "coupling_scale_errors": dict(msg.coupling_scale_errors),
            "global_rabi_error": msg.global_rabi_error,
            "timestamp": msg.timestamp,
            "metadata": dict(msg.metadata),
        }
    )


def delta_model_to_proto(model: HardwareDeltaModel) -> pb.HardwareDeltaModelProto:
    return pb.HardwareDeltaModelProto(
        device_id=model.device_id,
        substrate=model.substrate.value,
        drift=drift_to_proto(model.drift),
        n_sites=model.n_sites,
        metadata={k: str(v) for k, v in model.metadata.items()},
    )


def delta_model_from_proto(msg: pb.HardwareDeltaModelProto) -> HardwareDeltaModel:
    return HardwareDeltaModel.from_dict(
        {
            "device_id": msg.device_id,
            "substrate": msg.substrate,
            "drift": {
                "site_detuning_offsets": dict(msg.drift.site_detuning_offsets),
                "coupling_scale_errors": dict(msg.drift.coupling_scale_errors),
                "global_rabi_error": msg.drift.global_rabi_error,
                "timestamp": msg.drift.timestamp,
                "metadata": dict(msg.drift.metadata),
            },
            "n_sites": msg.n_sites,
            "metadata": dict(msg.metadata),
        }
    )

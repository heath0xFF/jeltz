"""Device discovery — scan profiles directory and instantiate adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jeltz.adapters.base import BaseAdapter
from jeltz.adapters.mock import MockAdapter
from jeltz.adapters.mqtt import MQTTAdapter
from jeltz.adapters.serial import SerialAdapter
from jeltz.devices.model import DeviceModel
from jeltz.profiles.parser import ProfileError, parse_profile

# Registry mapping protocol names to adapter classes.
_ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "mock": MockAdapter,
    "mqtt": MQTTAdapter,
    "serial": SerialAdapter,
}


def register_adapter(protocol: str, adapter_cls: type[BaseAdapter]) -> None:
    """Register an adapter class for a protocol name."""
    _ADAPTER_REGISTRY[protocol] = adapter_cls


def create_adapter(model: DeviceModel) -> BaseAdapter:
    """Create the appropriate adapter for a device model's protocol."""
    protocol = model.connection.protocol
    adapter_cls = _ADAPTER_REGISTRY.get(protocol)
    if adapter_cls is None:
        raise ValueError(
            f"unknown protocol: {protocol!r} (registered: {sorted(_ADAPTER_REGISTRY)})"
        )
    return adapter_cls(model.connection)


class DiscoveredDevice:
    """A parsed device profile paired with its adapter instance."""

    def __init__(self, model: DeviceModel, adapter: BaseAdapter) -> None:
        self.model = model
        self.adapter = adapter

    @property
    def name(self) -> str:
        return self.model.device.name


@dataclass
class DiscoveryResult:
    """Result of scanning a profiles directory."""

    devices: list[DiscoveredDevice] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)


def discover_profiles(profiles_dir: Path) -> DiscoveryResult:
    """Scan a directory for TOML profiles, parse each, and instantiate adapters.

    Skips files that fail to parse and collects errors.
    Returns both successfully discovered devices and any errors encountered.
    """
    result = DiscoveryResult()

    if not profiles_dir.is_dir():
        return result

    for path in sorted(profiles_dir.glob("**/*.toml")):
        try:
            model = parse_profile(path)
        except ProfileError as e:
            result.errors.append((path, str(e)))
            continue

        try:
            adapter = create_adapter(model)
        except ValueError as e:
            result.errors.append((path, str(e)))
            continue

        result.devices.append(DiscoveredDevice(model, adapter))

    return result

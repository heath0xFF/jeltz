"""Tests for device discovery — profile scanning and adapter instantiation."""

from __future__ import annotations

from pathlib import Path

import pytest

from jeltz.adapters.mock import MockAdapter
from jeltz.gateway.discovery import (
    DiscoveredDevice,
    create_adapter,
    discover_profiles,
)
from jeltz.profiles.parser import parse_profile

FIXTURES = Path(__file__).parents[1] / "fixtures"


class TestCreateAdapter:
    def test_creates_mock_adapter(self) -> None:
        model = parse_profile(FIXTURES / "ds18b20.toml")
        # Override protocol to mock so we can test without serial
        model.connection.protocol = "mock"
        adapter = create_adapter(model)
        assert isinstance(adapter, MockAdapter)

    def test_unknown_protocol_raises(self) -> None:
        model = parse_profile(FIXTURES / "ds18b20.toml")
        model.connection.protocol = "modbus"
        with pytest.raises(ValueError, match="unknown protocol.*modbus"):
            create_adapter(model)


class TestDiscoverProfiles:
    def test_discovers_toml_files(self, tmp_path: Path) -> None:
        profile = tmp_path / "sensor.toml"
        profile.write_text(
            '[device]\nname = "sensor"\n[connection]\nprotocol = "mock"\n'
        )
        result = discover_profiles(tmp_path)
        assert len(result.devices) == 1
        assert result.devices[0].name == "sensor"
        assert isinstance(result.devices[0].adapter, MockAdapter)
        assert result.errors == []

    def test_discovers_nested_profiles(self, tmp_path: Path) -> None:
        sub = tmp_path / "sensors"
        sub.mkdir()
        (sub / "temp.toml").write_text(
            '[device]\nname = "temp"\n[connection]\nprotocol = "mock"\n'
        )
        (sub / "humidity.toml").write_text(
            '[device]\nname = "humidity"\n[connection]\nprotocol = "mock"\n'
        )
        result = discover_profiles(tmp_path)
        assert len(result.devices) == 2
        names = {d.name for d in result.devices}
        assert names == {"temp", "humidity"}

    def test_reports_invalid_profiles(self, tmp_path: Path) -> None:
        (tmp_path / "good.toml").write_text(
            '[device]\nname = "good"\n[connection]\nprotocol = "mock"\n'
        )
        (tmp_path / "bad.toml").write_text("not valid toml {{{{")
        result = discover_profiles(tmp_path)
        assert len(result.devices) == 1
        assert result.devices[0].name == "good"
        assert len(result.errors) == 1

    def test_reports_unknown_protocol(self, tmp_path: Path) -> None:
        (tmp_path / "sensor.toml").write_text(
            '[device]\nname = "sensor"\n[connection]\nprotocol = "modbus"\n'
        )
        result = discover_profiles(tmp_path)
        assert len(result.devices) == 0
        assert len(result.errors) == 1
        assert "modbus" in result.errors[0][1]

    def test_nonexistent_directory_returns_empty(self, tmp_path: Path) -> None:
        result = discover_profiles(tmp_path / "nope")
        assert result.devices == []
        assert result.errors == []

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        result = discover_profiles(tmp_path)
        assert result.devices == []

    def test_ignores_non_toml_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("not a profile")
        (tmp_path / "sensor.toml").write_text(
            '[device]\nname = "sensor"\n[connection]\nprotocol = "mock"\n'
        )
        result = discover_profiles(tmp_path)
        assert len(result.devices) == 1


class TestDiscoveredDevice:
    def test_name_property(self) -> None:
        model = parse_profile(FIXTURES / "ds18b20.toml")
        model.connection.protocol = "mock"
        adapter = MockAdapter(model.connection)
        device = DiscoveredDevice(model, adapter)
        assert device.name == "ds18b20"

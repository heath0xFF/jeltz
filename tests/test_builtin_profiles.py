"""Tests for built-in device profiles.

Validates that all shipped profiles parse correctly, generate valid MCP tools,
and work together in a multi-device fleet scenario via mock adapters.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from jeltz.adapters.mock import MockAdapter
from jeltz.gateway.discovery import create_adapter, discover_profiles
from jeltz.gateway.server import JeltzServer
from jeltz.profiles.generator import generate_tools
from jeltz.profiles.parser import parse_profile

FIXTURES = Path(__file__).parent / "fixtures"
PROFILES = Path(__file__).parent.parent / "profiles"

# Every shipped profile (add new ones here as they're created)
SHIPPED_PROFILES = [
    "serial_sensor.toml",
    "mqtt_sensor.toml",
]

# Mock fixture for each shipped profile
MOCK_FIXTURES = [
    "serial_sensor_mock.toml",
    "mqtt_sensor_mock.toml",
]

# Simulated sensor responses for each mock device
MOCK_RESPONSES = {
    "serial_sensor": {
        "READ_TEMP": 22.1,
        "READ_HUMID": 58.7,
        "READ_ALL": "22.1,58.7",
        "STATUS": "OK",
        "PING": "PONG",
    },
    "mqtt_sensor": {
        "READ_TEMP": 21.8,
        "READ_HUMID": 65.3,
        "READ_ALL": "21.8,65.3",
        "PING": "PONG",
    },
}


class TestBuiltinProfilesParse:
    """All shipped profiles must parse without errors."""

    @pytest.mark.parametrize("profile_name", SHIPPED_PROFILES)
    def test_shipped_profile_parses(self, profile_name: str) -> None:
        model = parse_profile(PROFILES / profile_name)
        assert model.device.name
        assert model.device.description
        assert model.connection.protocol in ("serial", "mqtt")
        assert len(model.tools) >= 2

        # Every tool must have a command and a returns spec
        for tool in model.tools:
            assert tool.command, f"{tool.name} missing command"
            assert tool.returns, f"{tool.name} missing returns"

        # Health check should be configured
        assert model.health is not None
        assert model.health.check_command == "PING"
        assert model.health.expected == "PONG"

    @pytest.mark.parametrize("fixture_name", MOCK_FIXTURES)
    def test_mock_fixture_parses(self, fixture_name: str) -> None:
        model = parse_profile(FIXTURES / fixture_name)
        assert model.connection.protocol == "mock"

    def test_serial_sensor_tool_names(self) -> None:
        model = parse_profile(PROFILES / "serial_sensor.toml")
        names = {t.name for t in model.tools}
        assert names == {"get_temperature", "get_humidity", "get_reading", "get_status"}

    def test_mqtt_sensor_tool_names(self) -> None:
        model = parse_profile(PROFILES / "mqtt_sensor.toml")
        names = {t.name for t in model.tools}
        assert names == {"get_temperature", "get_humidity", "get_reading"}

    def test_serial_config(self) -> None:
        model = parse_profile(PROFILES / "serial_sensor.toml")
        assert model.connection.protocol == "serial"
        assert model.connection.baud_rate == 115200
        assert model.connection.timeout_ms == 3000

    def test_mqtt_config(self) -> None:
        model = parse_profile(PROFILES / "mqtt_sensor.toml")
        assert model.connection.protocol == "mqtt"
        assert model.connection.timeout_ms == 5000


class TestMockFixtureSync:
    """Mock fixtures must stay in sync with their real profiles."""

    @pytest.mark.parametrize("real_name,mock_name", [
        ("serial_sensor.toml", "serial_sensor_mock.toml"),
        ("mqtt_sensor.toml", "mqtt_sensor_mock.toml"),
    ])
    def test_tools_match_real_profile(self, real_name: str, mock_name: str) -> None:
        real = parse_profile(PROFILES / real_name)
        mock = parse_profile(FIXTURES / mock_name)

        assert real.device.name == mock.device.name
        assert real.device.description == mock.device.description
        assert len(real.tools) == len(mock.tools)

        for real_tool, mock_tool in zip(real.tools, mock.tools):
            assert real_tool.name == mock_tool.name, (
                f"Tool name mismatch: {real_tool.name} != {mock_tool.name}"
            )
            assert real_tool.command == mock_tool.command, (
                f"Command mismatch for {real_tool.name}: "
                f"{real_tool.command} != {mock_tool.command}"
            )
            assert real_tool.description == mock_tool.description, (
                f"Description mismatch for {real_tool.name}"
            )
            assert real_tool.returns == mock_tool.returns, (
                f"Returns mismatch for {real_tool.name}"
            )

    @pytest.mark.parametrize("real_name,mock_name", [
        ("serial_sensor.toml", "serial_sensor_mock.toml"),
        ("mqtt_sensor.toml", "mqtt_sensor_mock.toml"),
    ])
    def test_health_config_matches(self, real_name: str, mock_name: str) -> None:
        real = parse_profile(PROFILES / real_name)
        mock = parse_profile(FIXTURES / mock_name)

        assert real.health is not None
        assert mock.health is not None
        assert real.health.check_command == mock.health.check_command
        assert real.health.expected == mock.health.expected


class TestProfileSchemas:
    """Profiles generate correct MCP tool schemas."""

    def test_serial_sensor_tools(self) -> None:
        model = parse_profile(FIXTURES / "serial_sensor_mock.toml")
        tools = generate_tools(model)
        names = {t.name for t in tools}

        assert names == {
            "serial_sensor.get_temperature",
            "serial_sensor.get_humidity",
            "serial_sensor.get_reading",
            "serial_sensor.get_status",
        }

    def test_mqtt_sensor_tools(self) -> None:
        model = parse_profile(FIXTURES / "mqtt_sensor_mock.toml")
        tools = generate_tools(model)
        names = {t.name for t in tools}

        assert names == {
            "mqtt_sensor.get_temperature",
            "mqtt_sensor.get_humidity",
            "mqtt_sensor.get_reading",
        }

    def test_tool_descriptions_include_units(self) -> None:
        model = parse_profile(FIXTURES / "serial_sensor_mock.toml")
        tools = generate_tools(model)
        temp = next(t for t in tools if "temperature" in t.name)
        assert "celsius" in temp.description.lower()

        humid = next(t for t in tools if "humidity" in t.name)
        assert "percent" in humid.description.lower()


class TestProfileMockAdapters:
    """Mock variants can connect and respond to commands."""

    @pytest.mark.parametrize("fixture_name,device_name", [
        ("serial_sensor_mock.toml", "serial_sensor"),
        ("mqtt_sensor_mock.toml", "mqtt_sensor"),
    ])
    async def test_connect_and_health_check(
        self, fixture_name: str, device_name: str
    ) -> None:
        model = parse_profile(FIXTURES / fixture_name)
        adapter = create_adapter(model)
        assert isinstance(adapter, MockAdapter)

        adapter.responses = MOCK_RESPONSES[device_name]

        result = await adapter.connect()
        assert result.success

        result = await adapter.health_check()
        assert result.success

        await adapter.disconnect()

    @pytest.mark.parametrize("fixture_name,device_name", [
        ("serial_sensor_mock.toml", "serial_sensor"),
        ("mqtt_sensor_mock.toml", "mqtt_sensor"),
    ])
    async def test_tool_commands_return_expected_values(
        self, fixture_name: str, device_name: str
    ) -> None:
        model = parse_profile(FIXTURES / fixture_name)
        adapter = create_adapter(model)
        assert isinstance(adapter, MockAdapter)

        responses = MOCK_RESPONSES[device_name]
        adapter.responses = responses
        await adapter.connect()

        for tool in model.tools:
            assert tool.command is not None
            await adapter.send(tool.command)
            result = await adapter.receive()
            assert result.success, f"{tool.name}: {result.error}"
            assert result.data == responses[tool.command], (
                f"{tool.name}: expected {responses[tool.command]}, got {result.data}"
            )

        await adapter.disconnect()


@asynccontextmanager
async def fleet_env(
    fleet_dir: Path,
) -> AsyncGenerator[tuple, None]:
    """Start a JeltzServer with mock fleet and connect an MCP client."""
    server = JeltzServer(profiles_dir=fleet_dir, db_path=":memory:")
    discovery = await server.start()

    for device in discovery.devices:
        if isinstance(device.adapter, MockAdapter) and device.name in MOCK_RESPONSES:
            device.adapter.responses = MOCK_RESPONSES[device.name]

    try:
        async with create_connected_server_and_client_session(
            server._server,  # noqa: SLF001
            raise_exceptions=True,
        ) as client:
            yield client, server
    finally:
        await server.stop()


class TestFleetWithBuiltinProfiles:
    """Both mock devices running as a fleet through the MCP server."""

    @pytest.fixture
    def fleet_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "profiles"
        d.mkdir()
        for fixture in MOCK_FIXTURES:
            src = FIXTURES / fixture
            (d / fixture).write_text(src.read_text())
        return d

    async def test_fleet_discovers_all_devices(self, fleet_dir: Path) -> None:
        discovery = discover_profiles(fleet_dir)
        assert len(discovery.devices) == 2
        assert discovery.errors == []

        names = {d.name for d in discovery.devices}
        assert names == {"serial_sensor", "mqtt_sensor"}

    async def test_fleet_tool_listing(self, fleet_dir: Path) -> None:
        async with fleet_env(fleet_dir) as (client, _):
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}

            assert "serial_sensor.get_temperature" in names
            assert "mqtt_sensor.get_temperature" in names
            assert "fleet.list_devices" in names
            assert "fleet.get_all_readings" in names

            # 7 device tools + 4 fleet tools = 11
            assert len(tools.tools) == 11

    async def test_fleet_device_reads(self, fleet_dir: Path) -> None:
        async with fleet_env(fleet_dir) as (client, _):
            serial = await client.call_tool("serial_sensor.get_temperature", {})
            assert serial.isError is not True
            assert serial.structuredContent is not None
            assert serial.structuredContent["data"] == 22.1

            mqtt = await client.call_tool("mqtt_sensor.get_temperature", {})
            assert mqtt.isError is not True
            assert mqtt.structuredContent["data"] == 21.8

    async def test_fleet_list_devices(self, fleet_dir: Path) -> None:
        async with fleet_env(fleet_dir) as (client, _):
            result = await client.call_tool("fleet.list_devices", {})
            assert result.structuredContent is not None
            data = result.structuredContent
            assert data["count"] == 2

            for device in data["devices"]:
                assert device["connected"] is True
                assert len(device["tools"]) >= 2

    async def test_fleet_get_all_readings(self, fleet_dir: Path) -> None:
        async with fleet_env(fleet_dir) as (client, server):
            assert server.store is not None

            await server.store.record(
                "serial_sensor", "get_temperature", 22.1, "celsius"
            )
            await server.store.record(
                "mqtt_sensor", "get_temperature", 21.8, "celsius"
            )

            result = await client.call_tool("fleet.get_all_readings", {})
            assert result.structuredContent is not None
            assert result.structuredContent["count"] == 2

    async def test_fleet_get_history(self, fleet_dir: Path) -> None:
        async with fleet_env(fleet_dir) as (client, server):
            assert server.store is not None

            now = time.time()
            for i in range(5):
                await server.store.record(
                    "serial_sensor", "get_temperature", 22.0 + i * 0.1, "celsius",
                    timestamp=now - (i * 60),
                )

            result = await client.call_tool("fleet.get_history", {
                "device_id": "serial_sensor",
                "sensor_id": "get_temperature",
                "hours": 1,
            })
            assert result.isError is not True
            assert result.structuredContent is not None
            assert result.structuredContent["count"] == 5

    async def test_fleet_search_anomalies(self, fleet_dir: Path) -> None:
        async with fleet_env(fleet_dir) as (client, server):
            assert server.store is not None

            now = time.time()
            for i in range(10):
                await server.store.record(
                    "serial_sensor", "get_temperature", 22.0, "celsius",
                    timestamp=now - ((i + 1) * 3600),
                )
            # Outlier
            await server.store.record(
                "serial_sensor", "get_temperature", 85.0, "celsius",
                timestamp=now,
            )

            result = await client.call_tool("fleet.search_anomalies", {})
            assert result.structuredContent is not None
            assert result.structuredContent["count"] == 1
            assert (
                result.structuredContent["anomalies"][0]["device_id"] == "serial_sensor"
            )

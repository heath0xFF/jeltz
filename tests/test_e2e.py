"""End-to-end tests: TOML profile → JeltzServer → MCP client → tool call → response.

Uses the MCP SDK's in-memory transport to connect a real MCP client session
to a running JeltzServer. No stdio, no HTTP — just the full protocol over
memory streams. Mock adapters with configured responses simulate real devices.
"""

from __future__ import annotations

import random
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from jeltz.adapters.mock import MockAdapter
from jeltz.gateway.server import JeltzServer

TEMP_PROFILE = """\
[device]
name = "temp_sensor"
description = "Temperature sensor for fermentation tank 1"

[connection]
protocol = "mock"

[[tools]]
name = "get_reading"
description = "Get current temperature in Celsius"
command = "READ_TEMP"

[tools.returns]
type = "float"
unit = "celsius"

[health]
check_command = "PING"
expected = "PONG"
interval_ms = 10000
"""

PRESSURE_PROFILE = """\
[device]
name = "pressure_sensor"
description = "Pressure sensor on coolant line"

[connection]
protocol = "mock"

[[tools]]
name = "get_reading"
description = "Get current pressure in PSI"
command = "READ_PSI"

[tools.returns]
type = "float"
unit = "psi"

[[tools]]
name = "get_max"
description = "Get peak pressure since last reset"
command = "READ_MAX"

[tools.returns]
type = "float"
unit = "psi"
"""

MOCK_RESPONSES = {
    "temp_sensor": {"READ_TEMP": 22.5, "PING": "PONG"},
    "pressure_sensor": {"READ_PSI": 14.7, "READ_MAX": 18.3},
}


@pytest.fixture
def profiles_dir(tmp_path: Path) -> Path:
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "temp.toml").write_text(TEMP_PROFILE)
    (d / "pressure.toml").write_text(PRESSURE_PROFILE)
    return d


@asynccontextmanager
async def mcp_env(
    profiles_dir: Path,
) -> AsyncGenerator[tuple, None]:
    """Start a JeltzServer with mock devices and connect an MCP client.

    Must be used as `async with` inside a test function (not as a fixture)
    so that setup and teardown stay in the same anyio task.
    """
    server = JeltzServer(profiles_dir=profiles_dir, db_path=":memory:")
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


class TestToolListing:
    async def test_lists_all_device_tools(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.list_tools()
            names = {t.name for t in result.tools}

            assert "temp_sensor.get_reading" in names
            assert "pressure_sensor.get_reading" in names
            assert "pressure_sensor.get_max" in names

    async def test_lists_all_fleet_tools(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.list_tools()
            names = {t.name for t in result.tools}

            assert "fleet.list_devices" in names
            assert "fleet.get_all_readings" in names
            assert "fleet.get_history" in names
            assert "fleet.search_anomalies" in names

    async def test_total_tool_count(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.list_tools()
            # 3 device tools + 4 fleet tools = 7
            assert len(result.tools) == 7

    async def test_tool_has_schema(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.list_tools()
            tool = next(t for t in result.tools if t.name == "temp_sensor.get_reading")
            assert tool.inputSchema is not None
            assert tool.description


class TestDeviceToolCalls:
    async def test_call_device_tool_success(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("temp_sensor.get_reading", {})
            assert result.isError is not True
            assert len(result.content) > 0
            assert "22.5" in result.content[0].text

    async def test_device_tool_structured_content(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("temp_sensor.get_reading", {})
            assert result.structuredContent is not None
            assert result.structuredContent["data"] == 22.5

    async def test_call_device_tool_different_device(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("pressure_sensor.get_reading", {})
            assert result.isError is not True
            assert "14.7" in result.content[0].text

    async def test_call_second_tool_on_device(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("pressure_sensor.get_max", {})
            assert result.isError is not True
            assert "18.3" in result.content[0].text

    async def test_unknown_tool_returns_error(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("nonexistent.tool", {})
            assert result.isError is True
            assert "unknown tool" in result.content[0].text

    async def test_unknown_fleet_tool_returns_error(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("fleet.nonexistent", {})
            assert result.isError is True


class TestFleetListDevices:
    async def test_returns_all_devices(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("fleet.list_devices", {})
            assert result.isError is not True
            assert result.structuredContent is not None

            data = result.structuredContent
            assert data["count"] == 2
            names = {d["name"] for d in data["devices"]}
            assert names == {"temp_sensor", "pressure_sensor"}

    async def test_device_metadata(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("fleet.list_devices", {})
            data = result.structuredContent
            assert data is not None

            temp = next(d for d in data["devices"] if d["name"] == "temp_sensor")
            assert temp["protocol"] == "mock"
            assert temp["connected"] is True
            # healthy is False — connect_all doesn't run health checks
            assert temp["healthy"] is False
            assert "temp_sensor.get_reading" in temp["tools"]


class TestFleetGetAllReadings:
    async def test_empty_when_no_data(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, _):
            result = await client.call_tool("fleet.get_all_readings", {})
            assert result.isError is not True
            assert result.structuredContent is not None
            assert result.structuredContent["count"] == 0

    async def test_returns_stored_readings(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, server):
            assert server.store is not None
            await server.store.record("temp_sensor", "get_reading", 22.5, "celsius")
            await server.store.record("pressure_sensor", "get_reading", 14.7, "psi")

            result = await client.call_tool("fleet.get_all_readings", {})
            assert result.structuredContent is not None
            assert result.structuredContent["count"] == 2


class TestFleetGetHistory:
    async def test_returns_history(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, server):
            assert server.store is not None

            now = time.time()
            for i in range(5):
                await server.store.record(
                    "temp_sensor", "get_reading", 20.0 + i, "celsius",
                    timestamp=now - (i * 60),
                )

            result = await client.call_tool("fleet.get_history", {
                "device_id": "temp_sensor",
                "sensor_id": "get_reading",
                "hours": 1,
            })
            assert result.isError is not True
            assert result.structuredContent is not None

            data = result.structuredContent
            assert data["count"] == 5
            assert data["device_id"] == "temp_sensor"

    async def test_respects_limit(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, server):
            assert server.store is not None

            now = time.time()
            for i in range(10):
                await server.store.record(
                    "temp_sensor", "get_reading", 20.0 + i, "celsius",
                    timestamp=now - (i * 60),
                )

            result = await client.call_tool("fleet.get_history", {
                "device_id": "temp_sensor",
                "sensor_id": "get_reading",
                "hours": 1,
                "limit": 3,
            })
            assert result.structuredContent is not None
            assert result.structuredContent["count"] == 3


class TestFleetSearchAnomalies:
    async def test_no_anomalies_with_low_variance(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, server):
            assert server.store is not None

            # Small variance around 22.0 — latest reading is within normal range
            rng = random.Random(42)
            now = time.time()
            for i in range(20):
                await server.store.record(
                    "temp_sensor", "get_reading",
                    22.0 + rng.uniform(-0.1, 0.1), "celsius",
                    timestamp=now - (i * 3600),
                )

            result = await client.call_tool("fleet.search_anomalies", {})
            assert result.isError is not True
            assert result.structuredContent is not None
            assert result.structuredContent["count"] == 0

    async def test_detects_anomaly(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, server):
            assert server.store is not None

            now = time.time()
            for i in range(20):
                await server.store.record(
                    "temp_sensor", "get_reading", 22.0, "celsius",
                    timestamp=now - ((i + 1) * 3600),
                )
            # Outlier — way outside 2σ of a flat baseline
            await server.store.record(
                "temp_sensor", "get_reading", 99.0, "celsius",
                timestamp=now,
            )

            result = await client.call_tool("fleet.search_anomalies", {
                "threshold_sigma": 2.0,
            })
            assert result.structuredContent is not None
            data = result.structuredContent
            assert data["count"] == 1
            assert data["anomalies"][0]["device_id"] == "temp_sensor"


class TestMultiDeviceWorkflow:
    """Simulates a realistic multi-device query workflow."""

    async def test_discover_then_read_then_investigate(self, profiles_dir: Path) -> None:
        async with mcp_env(profiles_dir) as (client, server):
            assert server.store is not None

            # Step 1: Discover what's available
            result = await client.call_tool("fleet.list_devices", {})
            assert result.structuredContent is not None
            assert result.structuredContent["count"] == 2

            # Step 2: Read both devices directly
            temp = await client.call_tool("temp_sensor.get_reading", {})
            pressure = await client.call_tool("pressure_sensor.get_reading", {})
            assert temp.isError is not True
            assert pressure.isError is not True

            # Step 3: Store readings and query fleet snapshot
            await server.store.record("temp_sensor", "get_reading", 22.5, "celsius")
            await server.store.record("pressure_sensor", "get_reading", 14.7, "psi")

            readings = await client.call_tool("fleet.get_all_readings", {})
            assert readings.structuredContent is not None
            assert readings.structuredContent["count"] == 2

            # Step 4: Check for anomalies
            anomalies = await client.call_tool("fleet.search_anomalies", {})
            assert anomalies.isError is not True

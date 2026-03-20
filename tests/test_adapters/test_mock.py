"""Tests for the mock adapter."""

import pytest

from jeltz.adapters.mock import MockAdapter
from jeltz.devices.model import ConnectionConfig


@pytest.fixture
def mock_config() -> ConnectionConfig:
    return ConnectionConfig(protocol="mock", port=None, timeout_ms=1000)


@pytest.fixture
def adapter(mock_config: ConnectionConfig) -> MockAdapter:
    return MockAdapter(
        config=mock_config,
        responses={
            "READ_TEMP": 22.5,
            "PING": "PONG",
            "GET_ALL": [21.0, 22.5, 23.1],
        },
    )


class TestMockAdapterLifecycle:
    async def test_connect(self, adapter: MockAdapter):
        result = await adapter.connect()
        assert result.success
        assert adapter.connected

    async def test_disconnect(self, adapter: MockAdapter):
        await adapter.connect()
        result = await adapter.disconnect()
        assert result.success
        assert not adapter.connected

    async def test_send_while_disconnected(self, adapter: MockAdapter):
        result = await adapter.send("READ_TEMP")
        assert not result.success
        assert result.error == "not connected"

    async def test_receive_while_disconnected(self, adapter: MockAdapter):
        result = await adapter.receive()
        assert not result.success
        assert result.error == "not connected"


class TestMockAdapterCommands:
    async def test_send_and_receive(self, adapter: MockAdapter):
        await adapter.connect()
        send_result = await adapter.send("READ_TEMP")
        assert send_result.success

        recv_result = await adapter.receive()
        assert recv_result.success
        assert recv_result.data == 22.5

    async def test_send_tracks_history(self, adapter: MockAdapter):
        await adapter.connect()
        await adapter.send("READ_TEMP")
        await adapter.send("PING")

        assert adapter.send_history == ["READ_TEMP", "PING"]
        assert adapter.last_command == "PING"

    async def test_unknown_command(self, adapter: MockAdapter):
        await adapter.connect()
        await adapter.send("UNKNOWN")
        result = await adapter.receive()
        assert not result.success
        assert "unknown command" in (result.error or "")

    async def test_receive_without_send(self, adapter: MockAdapter):
        await adapter.connect()
        result = await adapter.receive()
        assert not result.success
        assert result.error == "no command sent"

    async def test_array_response(self, adapter: MockAdapter):
        await adapter.connect()
        await adapter.send("GET_ALL")
        result = await adapter.receive()
        assert result.success
        assert result.data == [21.0, 22.5, 23.1]


class TestMockAdapterHealth:
    async def test_healthy(self, adapter: MockAdapter):
        await adapter.connect()
        result = await adapter.health_check()
        assert result.success

    async def test_unhealthy(self, mock_config: ConnectionConfig):
        adapter = MockAdapter(config=mock_config, healthy=False)
        await adapter.connect()
        result = await adapter.health_check()
        assert not result.success
        assert result.error == "device unhealthy"

    async def test_health_check_while_disconnected(self, adapter: MockAdapter):
        result = await adapter.health_check()
        assert not result.success
        assert result.error == "not connected"

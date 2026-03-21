"""Tests for the MQTT adapter.

Uses a mocked paho-mqtt client — no broker needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest

from jeltz.adapters.mqtt import MQTTAdapter
from jeltz.devices.model import ConnectionConfig


@pytest.fixture
def mqtt_config() -> ConnectionConfig:
    return ConnectionConfig(
        protocol="mqtt",
        timeout_ms=2000,
        broker="localhost",
        mqtt_port=1883,
        topic_prefix="jeltz/test_device",
    )


def _make_mock_client() -> MagicMock:
    """Create a mock paho client that simulates successful connection."""
    from paho.mqtt.reasoncodes import ReasonCode

    client = MagicMock()
    client.is_connected.return_value = False
    client.on_connect = None
    client.on_message = None

    def fake_loop_start() -> None:
        # Simulate: connection established, trigger on_connect callback
        client.is_connected.return_value = True
        if client.on_connect:
            flags = MagicMock()
            rc = ReasonCode(mqtt.PacketTypes.CONNACK, "Success")
            client.on_connect(client, None, flags, rc, None)

    def fake_publish(topic: str, payload: object, **kwargs: object) -> MagicMock:
        info = MagicMock()
        info.rc = mqtt.MQTT_ERR_SUCCESS
        return info

    client.loop_start = MagicMock(side_effect=fake_loop_start)
    client.publish = MagicMock(side_effect=fake_publish)

    return client


@pytest.fixture
async def adapter(mqtt_config: ConnectionConfig) -> MQTTAdapter:
    """A connected MQTTAdapter with a mocked paho client."""
    mock_client = _make_mock_client()

    with patch(
        "jeltz.adapters.mqtt.mqtt.Client",
        return_value=mock_client,
    ):
        a = MQTTAdapter(mqtt_config)
        await a.connect()
        yield a
        await a.disconnect()


def _simulate_response(adapter: MQTTAdapter, payload: str) -> None:
    """Simulate a device publishing a response message."""
    if adapter._client and adapter._client.on_message:
        msg = MagicMock()
        msg.payload = payload.encode("utf-8")
        msg.topic = f"{adapter._topic_prefix}/response"
        adapter._client.on_message(adapter._client, None, msg)


class TestMQTTConnect:
    async def test_connect_success(self, mqtt_config: ConnectionConfig) -> None:
        mock_client = _make_mock_client()
        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client):
            adapter = MQTTAdapter(mqtt_config)
            result = await adapter.connect()
            assert result.success
            assert adapter.connected

            mock_client.connect.assert_called_once_with("localhost", 1883)
            mock_client.subscribe.assert_called_once_with("jeltz/test_device/response")
            await adapter.disconnect()

    async def test_connect_already_connected(self, adapter: MQTTAdapter) -> None:
        result = await adapter.connect()
        assert not result.success
        assert "already connected" in result.error

    async def test_connect_failure(self, mqtt_config: ConnectionConfig) -> None:
        mock_client = _make_mock_client()
        mock_client.connect.side_effect = ConnectionRefusedError("broker down")

        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client):
            adapter = MQTTAdapter(mqtt_config)
            result = await adapter.connect()
            assert not result.success
            assert "connection failed" in result.error

    async def test_connect_timeout(self, mqtt_config: ConnectionConfig) -> None:
        mock_client = _make_mock_client()
        # Don't trigger on_connect — simulate broker not responding
        mock_client.loop_start = MagicMock()  # Override: no callback trigger
        mock_client.is_connected.return_value = False

        mqtt_config_fast = ConnectionConfig(
            protocol="mqtt", timeout_ms=100,
            broker="localhost", mqtt_port=1883,
            topic_prefix="jeltz/test",
        )

        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client):
            adapter = MQTTAdapter(mqtt_config_fast)
            result = await adapter.connect()
            assert not result.success
            assert "timed out" in result.error
            assert not adapter.connected


class TestMQTTDisconnect:
    async def test_disconnect(self, adapter: MQTTAdapter) -> None:
        result = await adapter.disconnect()
        assert result.success
        assert not adapter.connected

    async def test_disconnect_when_not_connected(
        self, mqtt_config: ConnectionConfig
    ) -> None:
        adapter = MQTTAdapter(mqtt_config)
        result = await adapter.disconnect()
        assert result.success

    async def test_disconnect_handles_error(self, adapter: MQTTAdapter) -> None:
        adapter._client.disconnect.side_effect = OSError("broker gone")
        result = await adapter.disconnect()
        assert result.success
        assert not adapter.connected


class TestMQTTSend:
    async def test_send_string(self, adapter: MQTTAdapter) -> None:
        result = await adapter.send("READ_TEMP")
        assert result.success
        adapter._client.publish.assert_called_once_with(
            "jeltz/test_device/cmd", "READ_TEMP"
        )

    async def test_send_bytes(self, adapter: MQTTAdapter) -> None:
        payload = b"\x01\x02\x03"
        result = await adapter.send(payload)
        assert result.success
        adapter._client.publish.assert_called_once_with(
            "jeltz/test_device/cmd", payload
        )

    async def test_send_not_connected(self, mqtt_config: ConnectionConfig) -> None:
        adapter = MQTTAdapter(mqtt_config)
        result = await adapter.send("READ_TEMP")
        assert not result.success
        assert "not connected" in result.error

    async def test_send_publish_error(self, adapter: MQTTAdapter) -> None:
        info = MagicMock()
        info.rc = mqtt.MQTT_ERR_NO_CONN
        adapter._client.publish.side_effect = None
        adapter._client.publish.return_value = info

        result = await adapter.send("READ_TEMP")
        assert not result.success
        assert "publish failed" in result.error
        assert not adapter.connected

    async def test_send_exception_disconnects(self, adapter: MQTTAdapter) -> None:
        adapter._client.publish.side_effect = OSError("network down")

        result = await adapter.send("READ_TEMP")
        assert not result.success
        assert not adapter.connected


class TestMQTTReceive:
    async def test_receive_message(self, adapter: MQTTAdapter) -> None:
        # Simulate a response arriving
        _simulate_response(adapter, "22.5")

        result = await adapter.receive()
        assert result.success
        assert result.data == "22.5"

    async def test_receive_strips_whitespace(self, adapter: MQTTAdapter) -> None:
        _simulate_response(adapter, "  PONG  \r\n")

        result = await adapter.receive()
        assert result.success
        assert result.data == "PONG"

    async def test_receive_timeout(self, adapter: MQTTAdapter) -> None:
        # No message — should timeout
        result = await adapter.receive(timeout=0.05)
        assert not result.success
        assert "timed out" in result.error
        # Timeout is non-fatal
        assert adapter.connected

    async def test_receive_empty_response(self, adapter: MQTTAdapter) -> None:
        _simulate_response(adapter, "")

        result = await adapter.receive()
        assert not result.success
        assert "empty response" in result.error

    async def test_receive_not_connected(self, mqtt_config: ConnectionConfig) -> None:
        adapter = MQTTAdapter(mqtt_config)
        result = await adapter.receive()
        assert not result.success
        assert "not connected" in result.error


class TestMQTTHealthCheck:
    async def test_health_check_success(self, adapter: MQTTAdapter) -> None:
        # Schedule response before health check reads
        async def respond_soon() -> None:
            await asyncio.sleep(0.01)
            _simulate_response(adapter, "PONG")

        asyncio.create_task(respond_soon())

        result = await adapter.health_check()
        assert result.success
        assert result.data["response"] == "PONG"

    async def test_health_check_timeout(self, adapter: MQTTAdapter) -> None:
        # No response — health check should timeout
        # Override config for fast timeout
        adapter.config = ConnectionConfig(
            protocol="mqtt", timeout_ms=100,
            broker="localhost", mqtt_port=1883,
            topic_prefix="jeltz/test",
        )
        result = await adapter.health_check()
        assert not result.success
        assert "timed out" in result.error

    async def test_health_check_not_connected(
        self, mqtt_config: ConnectionConfig
    ) -> None:
        adapter = MQTTAdapter(mqtt_config)
        result = await adapter.health_check()
        assert not result.success
        assert "not connected" in result.error


class TestMQTTRoundTrip:
    """Simulates realistic command/response exchanges."""

    async def test_read_temp(self, adapter: MQTTAdapter) -> None:
        send = await adapter.send("READ_TEMP")
        assert send.success

        _simulate_response(adapter, "22.5")

        recv = await adapter.receive()
        assert recv.success
        assert recv.data == "22.5"

    async def test_multiple_commands(self, adapter: MQTTAdapter) -> None:
        await adapter.send("READ_TEMP")
        _simulate_response(adapter, "22.5")
        r1 = await adapter.receive()
        assert r1.data == "22.5"

        await adapter.send("READ_HUMID")
        _simulate_response(adapter, "61.2")
        r2 = await adapter.receive()
        assert r2.data == "61.2"

        await adapter.send("READ_ALL")
        _simulate_response(adapter, "22.5,61.2")
        r3 = await adapter.receive()
        assert r3.data == "22.5,61.2"


class TestMQTTDisconnectOnError:
    """Errors during send should transition to disconnected."""

    async def test_publish_error_disconnects(self, adapter: MQTTAdapter) -> None:
        adapter._client.publish.side_effect = OSError("lost")
        await adapter.send("READ_TEMP")
        assert not adapter.connected

    async def test_can_reconnect_after_error(
        self, mqtt_config: ConnectionConfig
    ) -> None:
        mock_client = _make_mock_client()

        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client):
            adapter = MQTTAdapter(mqtt_config)
            await adapter.connect()
            assert adapter.connected

            # Simulate error
            adapter._client.publish.side_effect = OSError("lost")
            await adapter.send("READ_TEMP")
            assert not adapter.connected

        # Reconnect with fresh client
        mock_client2 = _make_mock_client()
        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client2):
            result = await adapter.connect()
            assert result.success
            assert adapter.connected
            await adapter.disconnect()

    async def test_stale_messages_drained_on_error(
        self, mqtt_config: ConnectionConfig
    ) -> None:
        mock_client = _make_mock_client()

        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client):
            adapter = MQTTAdapter(mqtt_config)
            await adapter.connect()

            # Queue a response, then trigger an error before reading it
            _simulate_response(adapter, "stale_data")
            await asyncio.sleep(0)  # let call_soon_threadsafe enqueue
            adapter._client.publish.side_effect = OSError("lost")
            await adapter.send("READ_TEMP")
            assert not adapter.connected

        # Reconnect — stale message should be gone
        mock_client2 = _make_mock_client()
        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client2):
            await adapter.connect()
            # Receive should timeout, not return stale data
            result = await adapter.receive(timeout=0.05)
            assert not result.success
            assert "timed out" in result.error
            await adapter.disconnect()

    async def test_stale_messages_drained_on_disconnect(
        self, mqtt_config: ConnectionConfig
    ) -> None:
        mock_client = _make_mock_client()

        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client):
            adapter = MQTTAdapter(mqtt_config)
            await adapter.connect()

            _simulate_response(adapter, "stale_data")
            await asyncio.sleep(0)  # let call_soon_threadsafe enqueue
            await adapter.disconnect()

        # Reconnect — stale message should be gone
        mock_client2 = _make_mock_client()
        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client2):
            await adapter.connect()
            result = await adapter.receive(timeout=0.05)
            assert not result.success
            assert "timed out" in result.error
            await adapter.disconnect()


class TestMQTTBrokerDisconnect:
    """Broker-initiated disconnection should be detected."""

    async def test_on_disconnect_marks_adapter_disconnected(
        self, mqtt_config: ConnectionConfig
    ) -> None:
        from paho.mqtt.reasoncodes import ReasonCode

        mock_client = _make_mock_client()

        with patch("jeltz.adapters.mqtt.mqtt.Client", return_value=mock_client):
            adapter = MQTTAdapter(mqtt_config)
            await adapter.connect()
            assert adapter.connected

            # Simulate broker dropping the connection
            mock_client.is_connected.return_value = False
            if mock_client.on_disconnect:
                flags = MagicMock()
                rc = ReasonCode(mqtt.PacketTypes.DISCONNECT, "Normal disconnection")
                mock_client.on_disconnect(mock_client, None, flags, rc, None)

            assert not adapter.connected

"""Tests for the serial adapter.

Uses mocked asyncio streams — no hardware or serial port needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jeltz.adapters.serial import SerialAdapter
from jeltz.devices.model import ConnectionConfig


@pytest.fixture
def serial_config() -> ConnectionConfig:
    return ConnectionConfig(
        protocol="serial",
        port="/dev/ttyUSB0",
        baud_rate=115200,
        timeout_ms=2000,
    )


@pytest.fixture
def mock_streams():
    """Create mock reader/writer pair simulating a serial connection."""
    reader = AsyncMock(spec=asyncio.StreamReader)
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return reader, writer


@pytest.fixture
async def adapter(serial_config, mock_streams):
    """A connected SerialAdapter with mocked streams."""
    reader, writer = mock_streams
    with patch(
        "jeltz.adapters.serial.serial_asyncio.open_serial_connection",
        return_value=(reader, writer),
    ):
        a = SerialAdapter(serial_config)
        await a.connect()
        yield a
        await a.disconnect()


class TestSerialConnect:
    async def test_connect_success(self, serial_config, mock_streams) -> None:
        reader, writer = mock_streams
        with patch(
            "jeltz.adapters.serial.serial_asyncio.open_serial_connection",
            return_value=(reader, writer),
        ):
            adapter = SerialAdapter(serial_config)
            result = await adapter.connect()
            assert result.success
            assert adapter.connected

    async def test_connect_no_port(self) -> None:
        config = ConnectionConfig(protocol="serial", port=None, timeout_ms=1000)
        adapter = SerialAdapter(config)
        result = await adapter.connect()
        assert not result.success
        assert "no serial port" in result.error

    async def test_connect_already_connected(self, adapter) -> None:
        result = await adapter.connect()
        assert not result.success
        assert "already connected" in result.error

    async def test_connect_timeout(self, serial_config) -> None:
        async def slow_connect(**kwargs):
            await asyncio.sleep(10)

        with patch(
            "jeltz.adapters.serial.serial_asyncio.open_serial_connection",
            side_effect=slow_connect,
        ):
            adapter = SerialAdapter(serial_config)
            result = await adapter.connect()
            assert not result.success
            assert "timed out" in result.error

    async def test_connect_os_error(self, serial_config) -> None:
        with patch(
            "jeltz.adapters.serial.serial_asyncio.open_serial_connection",
            side_effect=OSError("Permission denied"),
        ):
            adapter = SerialAdapter(serial_config)
            result = await adapter.connect()
            assert not result.success
            assert "Permission denied" in result.error


class TestSerialDisconnect:
    async def test_disconnect(self, adapter, mock_streams) -> None:
        _, writer = mock_streams
        result = await adapter.disconnect()
        assert result.success
        assert not adapter.connected
        writer.close.assert_called_once()

    async def test_disconnect_when_not_connected(self, serial_config) -> None:
        adapter = SerialAdapter(serial_config)
        result = await adapter.disconnect()
        assert result.success

    async def test_disconnect_handles_close_error(self, adapter, mock_streams) -> None:
        _, writer = mock_streams
        writer.close.side_effect = OSError("port vanished")
        result = await adapter.disconnect()
        # Should still succeed — best-effort cleanup
        assert result.success
        assert not adapter.connected


class TestSerialSend:
    async def test_send_string(self, adapter, mock_streams) -> None:
        _, writer = mock_streams
        result = await adapter.send("READ_TEMP")
        assert result.success
        writer.write.assert_called_once_with(b"READ_TEMP\n")

    async def test_send_string_strips_existing_newline(self, adapter, mock_streams) -> None:
        _, writer = mock_streams
        result = await adapter.send("READ_TEMP\n")
        assert result.success
        writer.write.assert_called_once_with(b"READ_TEMP\n")

    async def test_send_bytes(self, adapter, mock_streams) -> None:
        _, writer = mock_streams
        result = await adapter.send(b"\x01\x02\x03")
        assert result.success
        writer.write.assert_called_once_with(b"\x01\x02\x03")

    async def test_send_not_connected(self, serial_config) -> None:
        adapter = SerialAdapter(serial_config)
        result = await adapter.send("READ_TEMP")
        assert not result.success
        assert "not connected" in result.error

    async def test_send_drain_error(self, adapter, mock_streams) -> None:
        _, writer = mock_streams
        writer.drain.side_effect = OSError("write failed")
        result = await adapter.send("READ_TEMP")
        assert not result.success
        assert "send failed" in result.error


class TestSerialReceive:
    async def test_receive_line(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(return_value=b"22.5\r\n")
        result = await adapter.receive()
        assert result.success
        assert result.data == "22.5"

    async def test_receive_strips_whitespace(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(return_value=b"  PONG  \r\n")
        result = await adapter.receive()
        assert result.success
        assert result.data == "PONG"

    async def test_receive_exact_length_returns_raw_bytes(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        payload = b"\x01\x02\x80\xff"
        reader.readexactly = AsyncMock(return_value=payload)
        result = await adapter.receive(length=4)
        assert result.success
        # Binary reads return raw bytes, not decoded text
        assert result.data == payload
        assert isinstance(result.data, bytes)

    async def test_receive_timeout(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(side_effect=asyncio.TimeoutError)
        result = await adapter.receive()
        assert not result.success
        assert "timed out" in result.error

    async def test_receive_custom_timeout(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams

        async def slow_read():
            await asyncio.sleep(10)

        reader.readline = AsyncMock(side_effect=slow_read)
        result = await adapter.receive(timeout=0.01)
        assert not result.success
        assert "timed out" in result.error

    async def test_receive_empty_response(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(return_value=b"\r\n")
        result = await adapter.receive()
        assert not result.success
        assert "empty response" in result.error

    async def test_receive_not_connected(self, serial_config) -> None:
        adapter = SerialAdapter(serial_config)
        result = await adapter.receive()
        assert not result.success
        assert "not connected" in result.error

    async def test_receive_incomplete_read(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readexactly = AsyncMock(
            side_effect=asyncio.IncompleteReadError(partial=b"\x01\x02", expected=4)
        )
        result = await adapter.receive(length=4)
        assert not result.success
        assert "incomplete" in result.error


class TestSerialDisconnectOnError:
    """I/O errors should transition the adapter to disconnected state."""

    async def test_send_error_disconnects(self, adapter, mock_streams) -> None:
        _, writer = mock_streams
        writer.drain.side_effect = OSError("port vanished")

        result = await adapter.send("READ_TEMP")
        assert not result.success
        assert not adapter.connected

    async def test_receive_error_disconnects(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(side_effect=OSError("device unplugged"))

        result = await adapter.receive()
        assert not result.success
        assert not adapter.connected

    async def test_incomplete_read_disconnects(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readexactly = AsyncMock(
            side_effect=asyncio.IncompleteReadError(partial=b"", expected=4)
        )
        result = await adapter.receive(length=4)
        assert not result.success
        assert not adapter.connected

    async def test_can_reconnect_after_error(self, serial_config, mock_streams) -> None:
        reader, writer = mock_streams
        with patch(
            "jeltz.adapters.serial.serial_asyncio.open_serial_connection",
            return_value=(reader, writer),
        ):
            adapter = SerialAdapter(serial_config)
            await adapter.connect()
            assert adapter.connected

            # Simulate I/O error
            writer.drain.side_effect = OSError("lost")
            await adapter.send("READ_TEMP")
            assert not adapter.connected

            # Reconnect
            writer.drain.side_effect = None
            result = await adapter.connect()
            assert result.success
            assert adapter.connected


class TestSerialHealthCheck:
    async def test_health_check_success(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(return_value=b"PONG\n")
        result = await adapter.health_check()
        assert result.success
        assert result.data["response"] == "PONG"

    async def test_health_check_timeout(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(side_effect=asyncio.TimeoutError)
        result = await adapter.health_check()
        assert not result.success
        assert "timed out" in result.error

    async def test_health_check_not_connected(self, serial_config) -> None:
        adapter = SerialAdapter(serial_config)
        result = await adapter.health_check()
        assert not result.success
        assert "not connected" in result.error


class TestSerialRoundTrip:
    """Simulates realistic command/response exchanges."""

    async def test_read_temp(self, adapter, mock_streams) -> None:
        reader, writer = mock_streams
        reader.readline = AsyncMock(return_value=b"22.5\n")

        send = await adapter.send("READ_TEMP")
        assert send.success
        writer.write.assert_called_with(b"READ_TEMP\n")

        recv = await adapter.receive()
        assert recv.success
        assert recv.data == "22.5"

    async def test_read_all(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(return_value=b"22.1,58.7\n")

        await adapter.send("READ_ALL")
        recv = await adapter.receive()
        assert recv.success
        assert recv.data == "22.1,58.7"

    async def test_multiple_commands(self, adapter, mock_streams) -> None:
        reader, _ = mock_streams
        reader.readline = AsyncMock(
            side_effect=[b"22.5\n", b"61.2\n", b"OK\n"]
        )

        await adapter.send("READ_TEMP")
        r1 = await adapter.receive()
        assert r1.data == "22.5"

        await adapter.send("READ_HUMID")
        r2 = await adapter.receive()
        assert r2.data == "61.2"

        await adapter.send("STATUS")
        r3 = await adapter.receive()
        assert r3.data == "OK"

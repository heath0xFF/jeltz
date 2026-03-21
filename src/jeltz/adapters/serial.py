"""Serial adapter — communicates with devices over UART/USB serial."""

from __future__ import annotations

import asyncio
import logging

import serial_asyncio

from jeltz.adapters.base import AdapterResult, BaseAdapter
from jeltz.devices.model import ConnectionConfig

logger = logging.getLogger(__name__)


class SerialAdapter(BaseAdapter):
    """Async serial adapter using pyserial-asyncio.

    Sends text commands terminated with newline and reads line-delimited
    responses. This matches the common pattern for microcontrollers
    (Arduino, ESP32, Pico) running a simple text command protocol.

    For binary reads, pass length to receive() — raw bytes are returned
    without text decoding.
    """

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    @property
    def connected(self) -> bool:
        return self._writer is not None

    def _mark_disconnected(self) -> None:
        """Transition to disconnected state (e.g. after I/O error)."""
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    async def connect(self) -> AdapterResult:
        if self.connected:
            return AdapterResult.fail("already connected")

        port = self.config.port
        if not port:
            return AdapterResult.fail("no serial port configured")

        baudrate = self.config.baud_rate or 9600
        timeout = self.config.timeout_ms / 1000

        try:
            self._reader, self._writer = await asyncio.wait_for(
                serial_asyncio.open_serial_connection(
                    url=port,
                    baudrate=baudrate,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            return AdapterResult.fail(f"connection timed out: {port}")
        except Exception as e:
            return AdapterResult.fail(f"connection failed: {e}")

        logger.info("connected to %s at %d baud", port, baudrate)
        return AdapterResult.ok()

    async def disconnect(self) -> AdapterResult:
        if not self.connected:
            return AdapterResult.ok()

        writer = self._writer
        self._reader = None
        self._writer = None

        try:
            if writer:
                writer.close()
                if hasattr(writer, "wait_closed"):
                    await writer.wait_closed()
        except Exception as e:
            logger.warning("error during disconnect: %s", e)

        return AdapterResult.ok()

    async def send(self, data: bytes | str) -> AdapterResult:
        if not self.connected or self._writer is None:
            return AdapterResult.fail("not connected")

        try:
            if isinstance(data, str):
                payload = (data.rstrip("\n") + "\n").encode("utf-8")
            else:
                payload = data

            self._writer.write(payload)
            await self._writer.drain()
        except Exception as e:
            self._mark_disconnected()
            return AdapterResult.fail(f"send failed: {e}")

        return AdapterResult.ok()

    async def receive(
        self, length: int | None = None, timeout: float | None = None
    ) -> AdapterResult:
        if not self.connected or self._reader is None:
            return AdapterResult.fail("not connected")

        if timeout is None:
            timeout = self.config.timeout_ms / 1000

        try:
            if length is not None:
                raw = await asyncio.wait_for(
                    self._reader.readexactly(length),
                    timeout=timeout,
                )
                # Binary read — return raw bytes, no text decoding
                return AdapterResult.ok(raw)
            else:
                raw = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=timeout,
                )
        except TimeoutError:
            # Timeout is non-fatal — the device may just be slow.
            # Don't disconnect; let the caller retry or escalate.
            return AdapterResult.fail("receive timed out")
        except asyncio.IncompleteReadError as e:
            self._mark_disconnected()
            return AdapterResult.fail(f"incomplete read: got {len(e.partial)} bytes")
        except Exception as e:
            self._mark_disconnected()
            return AdapterResult.fail(f"receive failed: {e}")

        # Text read — decode and strip line endings
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return AdapterResult.fail("empty response")

        return AdapterResult.ok(text)

    async def health_check(self) -> AdapterResult:
        if not self.connected:
            return AdapterResult.fail("not connected")

        send_result = await self.send("PING")
        if not send_result.success:
            return send_result

        recv_result = await self.receive()
        if not recv_result.success:
            return recv_result

        return AdapterResult.ok({"status": "healthy", "response": recv_result.data})

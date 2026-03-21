"""MQTT adapter — communicates with devices over MQTT pub/sub.

Uses paho-mqtt's background thread for network I/O, bridging messages
to asyncio via call_soon_threadsafe and an asyncio.Queue.

Topic layout (relative to topic_prefix in the connection config):
  {prefix}/cmd      — adapter publishes commands here
  {prefix}/response — adapter subscribes here for responses
"""

from __future__ import annotations

import asyncio
import logging

import paho.mqtt.client as mqtt

from jeltz.adapters.base import AdapterResult, BaseAdapter
from jeltz.devices.model import ConnectionConfig

logger = logging.getLogger(__name__)

# Max queued responses before dropping. Prevents unbounded memory growth
# if a device publishes faster than the gateway consumes.
_MAX_QUEUE_SIZE = 100


class MQTTAdapter(BaseAdapter):
    """Async MQTT adapter using paho-mqtt with asyncio bridge.

    Implements a request/reply pattern over MQTT: commands are published
    to {topic_prefix}/cmd, and responses are read from {topic_prefix}/response.
    """

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._client: mqtt.Client | None = None
        self._response_queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=_MAX_QUEUE_SIZE
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected_event: asyncio.Event | None = None

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.is_connected()

    @property
    def _broker(self) -> str:
        return getattr(self.config, "broker", "localhost")

    @property
    def _port(self) -> int:
        return int(getattr(self.config, "mqtt_port", 1883))

    @property
    def _topic_prefix(self) -> str:
        return getattr(self.config, "topic_prefix", "jeltz/device")

    @property
    def _cmd_topic(self) -> str:
        return f"{self._topic_prefix}/cmd"

    @property
    def _response_topic(self) -> str:
        return f"{self._topic_prefix}/response"

    def _drain_queue(self) -> None:
        """Discard any stale messages in the response queue."""
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _mark_disconnected(self) -> None:
        """Transition to disconnected state (e.g. after I/O error)."""
        client = self._client
        self._client = None
        self._loop = None
        if client is not None:
            try:
                client.disconnect()
                client.loop_stop()
            except Exception:
                pass
        self._drain_queue()

    async def connect(self) -> AdapterResult:
        if self.connected:
            return AdapterResult.fail("already connected")

        self._loop = asyncio.get_running_loop()
        self._connected_event = asyncio.Event()
        self._drain_queue()

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        timeout = self.config.timeout_ms / 1000

        def on_connect(
            client: mqtt.Client,
            userdata: object,
            flags: mqtt.ConnectFlags,
            rc: mqtt.ReasonCode,
            properties: mqtt.Properties | None,
        ) -> None:
            if not rc.is_failure:
                client.subscribe(self._response_topic)
                if self._loop and self._connected_event:
                    self._loop.call_soon_threadsafe(self._connected_event.set)

        def on_disconnect(
            client: mqtt.Client,
            userdata: object,
            flags: mqtt.DisconnectFlags,
            rc: mqtt.ReasonCode,
            properties: mqtt.Properties | None,
        ) -> None:
            logger.warning("MQTT broker disconnected: %s", rc)
            # Null out the client so connected returns False.
            # Don't call _mark_disconnected here — it calls loop_stop
            # which would deadlock from inside the network thread.
            self._client = None

        def on_message(
            client: mqtt.Client,
            userdata: object,
            msg: mqtt.MQTTMessage,
        ) -> None:
            text = msg.payload.decode("utf-8", errors="replace").strip()
            if self._loop:
                try:
                    self._loop.call_soon_threadsafe(
                        self._response_queue.put_nowait, text
                    )
                except Exception:
                    pass  # Queue full or loop closed

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message

        try:
            client.connect(self._broker, self._port)
            client.loop_start()
        except Exception as e:
            return AdapterResult.fail(f"connection failed: {e}")

        self._client = client

        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
        except TimeoutError:
            self._mark_disconnected()
            return AdapterResult.fail(
                f"connection timed out: {self._broker}:{self._port}"
            )

        logger.info("connected to MQTT broker %s:%d", self._broker, self._port)
        return AdapterResult.ok()

    async def disconnect(self) -> AdapterResult:
        if not self.connected:
            return AdapterResult.ok()

        client = self._client
        self._client = None
        self._loop = None

        try:
            if client:
                client.disconnect()
                client.loop_stop()
        except Exception as e:
            logger.warning("error during disconnect: %s", e)

        self._drain_queue()
        return AdapterResult.ok()

    async def send(self, data: bytes | str) -> AdapterResult:
        if not self.connected or self._client is None:
            return AdapterResult.fail("not connected")

        # Paho accepts both str and bytes payloads natively
        payload = data

        try:
            info = self._client.publish(self._cmd_topic, payload)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                self._mark_disconnected()
                return AdapterResult.fail(f"publish failed: rc={info.rc}")
        except Exception as e:
            self._mark_disconnected()
            return AdapterResult.fail(f"send failed: {e}")

        return AdapterResult.ok()

    async def receive(
        self, length: int | None = None, timeout: float | None = None
    ) -> AdapterResult:
        if not self.connected:
            return AdapterResult.fail("not connected")

        if timeout is None:
            timeout = self.config.timeout_ms / 1000

        try:
            text = await asyncio.wait_for(
                self._response_queue.get(), timeout=timeout
            )
        except TimeoutError:
            # Timeout is non-fatal — the device may just be slow.
            return AdapterResult.fail("receive timed out")
        except Exception as e:
            self._mark_disconnected()
            return AdapterResult.fail(f"receive failed: {e}")

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

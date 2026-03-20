"""Base adapter interface and result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jeltz.devices.model import ConnectionConfig


@dataclass
class AdapterResult:
    """Wraps adapter operation outcomes. Adapters never raise — they return these."""

    success: bool
    data: Any = None
    error: str | None = None

    @classmethod
    def ok(cls, data: Any = None) -> AdapterResult:
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> AdapterResult:
        return cls(success=False, error=error)


class BaseAdapter(ABC):
    """Abstract base for all protocol adapters."""

    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config

    @abstractmethod
    async def connect(self) -> AdapterResult: ...

    @abstractmethod
    async def disconnect(self) -> AdapterResult: ...

    @abstractmethod
    async def send(self, data: bytes | str) -> AdapterResult: ...

    @abstractmethod
    async def receive(
        self, length: int | None = None, timeout: float | None = None
    ) -> AdapterResult: ...

    @abstractmethod
    async def health_check(self) -> AdapterResult: ...

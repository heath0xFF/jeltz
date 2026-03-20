"""Integration test: parse a profile → create adapter → execute a tool call."""

from pathlib import Path

from jeltz.adapters.mock import MockAdapter
from jeltz.profiles.parser import parse_profile

FIXTURES = Path(__file__).parent / "fixtures"


async def test_profile_to_mock_tool_call():
    """End-to-end: parse ds18b20 profile, wire up mock adapter, execute tool."""
    model = parse_profile(FIXTURES / "ds18b20.toml")

    # Create a mock adapter with responses matching the profile's commands
    adapter = MockAdapter(
        config=model.connection,
        responses={
            "READ_TEMP": 22.5,
            "GET_RES": 12,
            "PING": "PONG",
        },
    )

    await adapter.connect()
    assert adapter.connected

    # Execute each tool by sending its command
    for tool in model.tools:
        assert tool.command is not None
        send_result = await adapter.send(tool.command)
        assert send_result.success

        recv_result = await adapter.receive()
        assert recv_result.success
        assert recv_result.data is not None

    # Verify health check works with the profile's expected values
    assert model.health is not None
    await adapter.send(model.health.check_command)
    health_result = await adapter.receive()
    assert health_result.success
    assert health_result.data == model.health.expected

    await adapter.disconnect()
    assert not adapter.connected

"""Tests for the TOML profile parser."""

from pathlib import Path

import pytest

from jeltz.profiles.parser import ProfileError, parse_profile, parse_profile_string

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestParseProfile:
    def test_parse_ds18b20(self):
        model = parse_profile(FIXTURES / "ds18b20.toml")

        assert model.device.name == "ds18b20"
        assert model.device.description == "Dallas 1-Wire temperature sensor via serial bridge"
        assert model.device.handler is None

        assert model.connection.protocol == "serial"
        assert model.connection.port == "/dev/ttyUSB0"
        assert model.connection.baud_rate == 9600
        assert model.connection.timeout_ms == 2000

        assert len(model.tools) == 2

        read_tool = model.tools[0]
        assert read_tool.name == "get_reading"
        assert read_tool.command == "READ_TEMP"
        assert read_tool.returns is not None
        assert read_tool.returns.type == "float"
        assert read_tool.returns.unit == "celsius"

        res_tool = model.tools[1]
        assert res_tool.name == "get_resolution"
        assert res_tool.command == "GET_RES"

        assert model.health is not None
        assert model.health.check_command == "PING"
        assert model.health.expected == "PONG"
        assert model.health.interval_ms == 10000

    def test_file_not_found(self):
        with pytest.raises(ProfileError, match="profile not found"):
            parse_profile(Path("/nonexistent/sensor.toml"))

    def test_minimal_profile(self):
        toml = """
[device]
name = "minimal"

[connection]
protocol = "mock"
"""
        model = parse_profile_string(toml)
        assert model.device.name == "minimal"
        assert model.tools == []
        assert model.health is None

    def test_invalid_toml(self):
        with pytest.raises(ProfileError, match="invalid TOML"):
            parse_profile_string("this is not [valid toml")

    def test_missing_device_section(self):
        toml = """
[connection]
protocol = "serial"
"""
        with pytest.raises(ProfileError, match="invalid profile"):
            parse_profile_string(toml)

    def test_missing_connection_section(self):
        toml = """
[device]
name = "no_connection"
"""
        with pytest.raises(ProfileError, match="invalid profile"):
            parse_profile_string(toml)

    def test_tool_with_params(self):
        toml = """
[device]
name = "parameterized"

[connection]
protocol = "serial"

[[tools]]
name = "get_reading"
description = "Read a specific channel"

[tools.params.channel]
type = "int"
min = 0
max = 7
description = "ADC channel index"

[tools.returns]
type = "float"
unit = "volts"
"""
        model = parse_profile_string(toml)
        tool = model.tools[0]
        assert "channel" in tool.params
        param = tool.params["channel"]
        assert param.type == "int"
        assert param.min == 0
        assert param.max == 7

    def test_profile_with_handler(self):
        toml = """
[device]
name = "custom"
handler = "handlers.custom_sensor"

[connection]
protocol = "serial"
"""
        model = parse_profile_string(toml)
        assert model.device.handler == "handlers.custom_sensor"

    def test_extra_connection_fields_allowed(self):
        toml = """
[device]
name = "mqtt_device"

[connection]
protocol = "mqtt"
broker = "localhost"
topic = "sensors/temp"
qos = 1
"""
        model = parse_profile_string(toml)
        assert model.connection.protocol == "mqtt"
        # Extra fields accessible via model_extra
        assert model.connection.model_extra is not None
        assert model.connection.model_extra["broker"] == "localhost"
        assert model.connection.model_extra["topic"] == "sensors/temp"

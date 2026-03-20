"""Jeltz CLI."""

import click


@click.group()
@click.version_option()
def main() -> None:
    """Jeltz — MCP gateway for physical devices."""

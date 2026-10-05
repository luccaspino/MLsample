"""Backward-compatible command-line alias."""

from ic.main import app


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()

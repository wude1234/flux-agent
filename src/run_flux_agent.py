"""Convenience entrypoint for the FLUX-backed T2I agent."""

from __future__ import annotations

from .run_m4 import main


def _with_default_generator(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        return None
    if "--generator" in argv:
        return argv
    return ["--generator", "flux", *argv]


if __name__ == "__main__":
    import sys

    raise SystemExit(main(_with_default_generator(sys.argv[1:])))

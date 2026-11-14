"""Command line interface of ``lrmlp`` (placeholder).

Only ``--version`` and ``--help`` work so far. The subcommands follow later.
"""

from __future__ import annotations

import sys

from lrmlp import __version__

_HELP = """\
usage: lrmlp [--version] [--help] <command> [<args>]

Machine learning potentials with long-range interaction.

options:
  --version   show the program version and exit
  -h, --help  show this help message and exit

No subcommands are implemented yet.
"""


def main(argv: list[str] | None = None) -> int:
    """Run the ``lrmlp`` command line interface.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: ``0`` for ``--version`` and ``--help``, ``2`` otherwise.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--version"]:
        print(f"lrmlp {__version__}")
        return 0
    if args and args[0] in ("-h", "--help"):
        print(_HELP, end="")
        return 0
    print("lrmlp: not implemented yet (see 'lrmlp --help')", file=sys.stderr)
    return 2

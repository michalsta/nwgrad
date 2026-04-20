import importlib.metadata
from pathlib import Path


__version__ = importlib.metadata.version("nwgrad")


def module_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="nwgrad: A module that prints Hello World message."
    )
    parser.add_argument("--version", "-v", action="version", version=__version__)
    parser.add_argument(
        "--include",
        "-i",
        help="Print include path for C++ headers",
        action="store_true",
    )

    args = parser.parse_args()

    if args.include:
        print(Path(__file__).parent / "cpp")
    else:
        parser.print_help()


if __name__ == "__main__":
    module_main()
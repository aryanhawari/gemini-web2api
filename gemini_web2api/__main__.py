"""CLI entrypoint: python -m gemini_web2api"""

import sys

from .config import apply_args, load_config, parse_args
from .server import serve


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    apply_args(config, args)
    serve(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())

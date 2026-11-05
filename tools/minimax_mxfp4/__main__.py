import argparse


def main():
    parser = argparse.ArgumentParser(description='MiniMax packed MXFP4 conversion and evaluation')
    parser.add_argument('command', choices=['convert', 'run'])
    parser.add_argument('--source')
    parser.add_argument('--output')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--run')
    args = parser.parse_args()
    if args.command == 'convert':
        if not args.source or not args.output or args.threads < 1:
            parser.error('convert requires --source, --output, and positive --threads')
        from .convert import convert
        convert(args.source, args.output, args.threads)
    else:
        if not args.run:
            parser.error('run requires --run')
        from .workflow import run
        run(args.run)


if __name__ == '__main__':
    main()

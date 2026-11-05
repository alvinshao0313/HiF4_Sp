import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['download', 'prepare', 'run'])
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    if args.command == 'download':
        from .download import download
        download(args.run)
    elif args.command == 'prepare':
        from .data import prepare
        prepare(args.run)
    else:
        from .workflow import run
        run(args.run)


if __name__ == '__main__':
    main()

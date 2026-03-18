import json
from globalGB.GB_runner import GBSearchRunner
from globalGB.search_utils_GB import GBConfig

def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Search for Galactic binaries in Mojito data.")
    parser.add_argument("which_run", type=str, choices=["even1st", "even", "odd"], help="Window set to analyze.")
    parser.add_argument("batch_index", type=int, help="Batch index of frequency windows to process.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open('globalGB/GB_search_config.json', 'r') as f:
        config = json.load(f)
        config = GBConfig(config)
    runner = GBSearchRunner(
        batch_index=args.batch_index,
        which_run=args.which_run,
        config=config,
    )
    runner.run()


if __name__ == "__main__":
    main()

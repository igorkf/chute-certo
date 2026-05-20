"""Download historical Brasileirão Série A fixtures from API-Football."""

import argparse

from chute_certo.ingestion.api_football import Settings, fetch_fixtures, save_fixtures

DEFAULT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    parser = argparse.ArgumentParser(description="Download Brasileirão match data")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to download (default: {DEFAULT_SEASONS})",
    )
    args = parser.parse_args()

    settings = Settings()

    for season in args.seasons:
        fixtures = fetch_fixtures(season, settings)
        save_fixtures(fixtures, season)


if __name__ == "__main__":
    main()

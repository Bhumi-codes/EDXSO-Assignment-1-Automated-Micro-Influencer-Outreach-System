"""Validated settings for the YouTube collection stage.

Run with `uv run --no-project config.py` to check local configuration.
Entry points load local .env settings via python-dotenv and make no API requests.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Mapping


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = PROJECT_DIR / 'data' / 'outreach.db'


def load_environment(path: Path | None = None) -> None:
    """Load a local file without replacing process variables or expanding values."""
    from dotenv import load_dotenv
    load_dotenv(path or PROJECT_DIR / '.env', override=False, interpolate=False)


def database_path(env: Mapping[str, str] | None = None) -> Path:
    """Return one default for every stage; no key, DB access or migration needed."""
    source = os.environ if env is None else env
    raw = source.get('DATABASE_PATH', '').strip()
    if not raw:
        return DEFAULT_DATABASE_PATH
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


DEFAULT_QUERIES = (
    "Python programming and development",
    "artificial intelligence with Python",
    "machine learning with Python",
    "deep learning with Python",
    "Python data science",
    "Python data analysis",
    "data structures and algorithms in Python",
    "Python automation scripting",
    "Django Python web development",
    "Flask Python web development",
    "FastAPI Python backend development",
    "computer vision with Python",
)


class ConfigurationError(ValueError):
    """A setting is missing or invalid; messages never include secret values."""


def _integer(
    env: Mapping[str, str], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw = env.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be a whole number.") from None
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{name} must be between {minimum} and {maximum}, inclusive."
        )
    return value


@dataclass(frozen=True)
class Settings:
    youtube_api_key: str = field(repr=False)
    discovery_target: int  # Number of size-eligible channels to collect per run.
    recent_video_limit: int
    min_subscribers: int
    max_subscribers: int
    search_queries: tuple[str, ...]
    max_search_requests: int
    max_playlist_pages_per_channel: int
    request_timeout_seconds: int
    data_dir: Path
    database_override: Path | None = None

    @property
    def database_path(self) -> Path:
        return self.database_override or self.data_dir / "outreach.db"

    def public_summary(self) -> dict[str, object]:
        """Return an explicit allowlist of safe settings for display."""
        return {
            "youtube_api_key": "configured (hidden)",
            "discovery_target": self.discovery_target,
            "recent_video_limit": self.recent_video_limit,
            "subscriber_range": [self.min_subscribers, self.max_subscribers],
            "search_queries": list(self.search_queries),
            "max_search_requests": self.max_search_requests,
            "max_playlist_pages_per_channel": self.max_playlist_pages_per_channel,
            "request_timeout_seconds": self.request_timeout_seconds,
            "database_path": str(self.database_path),
        }


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load local .env for process settings; explicit mappings remain isolated."""
    if env is None:
        load_environment()
    source = os.environ if env is None else env
    key = source.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        raise ConfigurationError("YOUTUBE_API_KEY is missing or empty.")
    if any(character.isspace() for character in key):
        raise ConfigurationError("YOUTUBE_API_KEY must not contain whitespace.")

    minimum = _integer(source, "MIN_SUBSCRIBERS", 5000, 0, 1_000_000_000)
    maximum = _integer(source, "MAX_SUBSCRIBERS", 100000, 0, 1_000_000_000)
    if minimum > maximum:
        raise ConfigurationError("MIN_SUBSCRIBERS must not exceed MAX_SUBSCRIBERS.")

    queries = tuple(
        query.strip()
        for query in source.get(
            "YOUTUBE_SEARCH_QUERIES", "|".join(DEFAULT_QUERIES)
        ).split("|")
        if query.strip()
    )
    queries = tuple(dict.fromkeys(queries))
    if not queries:
        raise ConfigurationError("YOUTUBE_SEARCH_QUERIES needs at least one query.")

    return Settings(
        youtube_api_key=key,
        discovery_target=_integer(source, "DISCOVERY_TARGET", 50, 1, 5000),
        recent_video_limit=_integer(source, "RECENT_VIDEO_LIMIT", 10, 1, 10),
        min_subscribers=minimum,
        max_subscribers=maximum,
        search_queries=queries,
        max_search_requests=_integer(source, "MAX_SEARCH_REQUESTS", 12, 1, 1000),
        max_playlist_pages_per_channel=_integer(
            source, "MAX_PLAYLIST_PAGES_PER_CHANNEL", 3, 1, 100
        ),
        request_timeout_seconds=_integer(
            source, "REQUEST_TIMEOUT_SECONDS", 30, 1, 300
        ),
        data_dir=PROJECT_DIR / "data",
        database_override=database_path(source),
    )


def main() -> int:
    try:
        settings = load_settings()
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    print("Configuration OK (local validation only; API access not tested).")
    print(json.dumps(settings.public_summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

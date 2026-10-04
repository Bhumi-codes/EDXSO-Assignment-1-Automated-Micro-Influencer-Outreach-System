"""Discover YouTube channels until the configured size-eligible target is met.

Normal execution makes live API requests. --self-check uses synthetic responses
and a disposable database, with no network requests or real creator data.
"""

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import TemporaryDirectory
from typing import Any

import httpx
from pydantic import ValidationError

from config import ConfigurationError, Settings, load_settings
from database import (
    get_channel, open_database, record_counts, save_channel, save_size_result,
)
from schemas import ChannelRecord, SizeEligibilityResult


API_BASE = "https://www.googleapis.com/youtube/v3/"
CHANNEL_ID_PATTERN = re.compile(r"UC[A-Za-z0-9_-]{22}")


class YouTubeRequestError(RuntimeError):
    """Safe API error messages: never include request headers or raw responses."""


@dataclass
class DiscoveryReport:
    channels: list[ChannelRecord] = field(default_factory=list)
    size_results: list[SizeEligibilityResult] = field(default_factory=list)
    unresolved: dict[str, str] = field(default_factory=dict)
    search_calls: int = 0
    channel_calls: int = 0
    saved_records: int = 0
    invalid_search_items: int = 0
    candidate_ids_seen: int = 0
    stopped_because: str = ""
    error: str | None = None

    @property
    def size_pass_count(self) -> int:
        return sum(result.status == "PASS" for result in self.size_results)


def _api_error(response: httpx.Response) -> str:
    hints = {
        "keyInvalid": "Check the YouTube API key.",
        "API_KEY_INVALID": "Check the YouTube API key.",
        "accessNotConfigured": "Enable YouTube Data API v3 in the key's project.",
        "SERVICE_DISABLED": "Enable YouTube Data API v3 in the key's project.",
        "quotaExceeded": "Check the project's YouTube quota in Google Cloud.",
        "dailyLimitExceeded": "Check the project's YouTube quota in Google Cloud.",
        "rateLimitExceeded": "The request rate limit was reached; no retry was made.",
        "forbidden": "Check the key's API and application restrictions.",
        "API_KEY_SERVICE_BLOCKED": "Allow YouTube Data API v3 in the key restrictions.",
        "API_KEY_IP_ADDRESS_BLOCKED": "Check the key's allowed public IP addresses.",
    }
    reasons: list[str] = []
    try:
        body = response.json()
        error = body.get("error", {}) if isinstance(body, dict) else {}
        if isinstance(error, dict):
            for entry in error.get("errors", []) + error.get("details", []):
                if isinstance(entry, dict) and isinstance(entry.get("reason"), str):
                    reasons.append(entry["reason"])
    except (ValueError, TypeError):
        pass
    hint = next((hints[reason] for reason in reasons if reason in hints), None)
    if hint is None:
        if response.status_code in (400, 401, 403):
            hint = "Check API enablement, key validity, and key restrictions."
        elif response.status_code == 429:
            hint = "Check quota and request rate; no retry was made."
        elif response.status_code >= 500:
            hint = "YouTube reported a server error; no retry was made."
        else:
            hint = "YouTube rejected the request; no retry was made."
    return f"YouTube HTTP {response.status_code}. {hint}"


def _request(client: httpx.Client, endpoint: str, parameters: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.get(API_BASE + endpoint, params=parameters)
    except httpx.TimeoutException:
        raise YouTubeRequestError("YouTube request timed out; no retry was made.") from None
    except httpx.RequestError:
        raise YouTubeRequestError("Could not reach YouTube; check your connection.") from None
    if not response.is_success:
        raise YouTubeRequestError(_api_error(response)) from None
    try:
        body = response.json()
    except ValueError:
        raise YouTubeRequestError("YouTube returned a response that was not valid JSON.") from None
    if not isinstance(body, dict) or not isinstance(body.get("items"), list):
        raise YouTubeRequestError("YouTube returned an unexpected response structure.")
    return body


def _channel_record(item: dict[str, Any], query: str, observed_at: datetime) -> ChannelRecord:
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    details = item.get("contentDetails", {})
    if not all(isinstance(value, dict) for value in (snippet, statistics, details)):
        raise ValueError("Invalid channel sections.")
    related = details.get("relatedPlaylists", {})
    if not isinstance(related, dict):
        raise ValueError("Invalid playlist section.")
    return ChannelRecord(
        channel_id=item["id"],
        title=snippet.get("title"),
        profile_url=f"https://www.youtube.com/channel/{item['id']}",
        description=snippet.get("description"),
        subscriber_count=statistics.get("subscriberCount"),
        hidden_subscriber_count=statistics.get("hiddenSubscriberCount"),
        country=snippet.get("country"),
        uploads_playlist_id=related.get("uploads"),
        discovery_query=query,
        collected_at=observed_at,
    )


def evaluate_size(channel: ChannelRecord, settings: Settings) -> SizeEligibilityResult:
    """Apply only the inclusive subscriber-size rule, never campaign relevance."""
    count = channel.subscriber_count
    if channel.hidden_subscriber_count is True:
        status = "NEEDS_REVIEW"
        reason = "Subscriber count is hidden; size eligibility is unknown."
    elif count is None:
        status = "NEEDS_REVIEW"
        reason = "Subscriber count is unavailable; size eligibility is unknown."
    elif settings.min_subscribers <= count <= settings.max_subscribers:
        status = "PASS"
        reason = (
            f"Reported subscribers {count:,} are within the inclusive range "
            f"{settings.min_subscribers:,}-{settings.max_subscribers:,}."
        )
    else:
        status = "FAIL"
        reason = (
            f"Reported subscribers {count:,} are outside the inclusive range "
            f"{settings.min_subscribers:,}-{settings.max_subscribers:,}."
        )
    return SizeEligibilityResult(
        channel_id=channel.channel_id, status=status, reasons=[reason],
        evaluated_at=datetime.now(timezone.utc),
    )


def discover(settings: Settings, client: httpx.Client, database_path: Path) -> DiscoveryReport:
    """DISCOVERY_TARGET means size-eligible channels collected in this run.

    Query pages are visited in rotation. A page's channels are inspected in search
    order, while channel response order is ignored. Each candidate is tried once.
    Completed batches commit before the next network request. Every valid record
    in a requested batch is saved, so size passes can exceed the minimum target.
    """
    report = DiscoveryReport()
    pages = deque((query, None) for query in settings.search_queries)
    visited_pages: set[tuple[str, str | None]] = set()
    seen: set[str] = set()
    with open_database(database_path):
        pass
    try:
        while (
            pages and report.search_calls < settings.max_search_requests
            and report.size_pass_count < settings.discovery_target
        ):
            query, token = pages.popleft()
            if (query, token) in visited_pages:
                continue
            visited_pages.add((query, token))
            parameters: dict[str, Any] = {
                "part": "snippet", "type": "video", "q": query,
                "order": "relevance", "maxResults": 50,
                "fields": "nextPageToken,items(snippet/channelId)",
            }
            if token is not None:
                parameters["pageToken"] = token
            report.search_calls += 1
            search = _request(client, "search", parameters)
            next_token = search.get("nextPageToken")
            if next_token is not None and not isinstance(next_token, str):
                raise YouTubeRequestError("YouTube returned an invalid search page token.")
            if next_token:
                pages.append((query, next_token))
            candidates: list[str] = []
            for item in search["items"]:
                snippet = item.get("snippet") if isinstance(item, dict) else None
                channel_id = snippet.get("channelId") if isinstance(snippet, dict) else None
                if not isinstance(channel_id, str) or not CHANNEL_ID_PATTERN.fullmatch(channel_id):
                    report.invalid_search_items += 1
                    continue
                if channel_id not in seen:
                    seen.add(channel_id)
                    candidates.append(channel_id)
            report.candidate_ids_seen = len(seen)
            while candidates and report.size_pass_count < settings.discovery_target:
                batch_size = 50
                batch, candidates = candidates[:batch_size], candidates[batch_size:]
                report.channel_calls += 1
                try:
                    response = _request(client, "channels", {
                        "part": "snippet,statistics,contentDetails",
                        "id": ",".join(batch), "maxResults": 50,
                        "fields": "items(id,snippet(title,description,country),statistics(subscriberCount,hiddenSubscriberCount),contentDetails/relatedPlaylists/uploads)",
                    })
                except YouTubeRequestError:
                    for identifier in batch:
                        report.unresolved[identifier] = "Channel-detail request failed."
                    raise
                by_id: dict[str, dict[str, Any]] = {}
                duplicate_ids: set[str] = set()
                for item in response["items"]:
                    identifier = item.get("id") if isinstance(item, dict) else None
                    if isinstance(identifier, str) and identifier in batch:
                        if identifier in by_id:
                            duplicate_ids.add(identifier)
                        by_id[identifier] = item
                records: list[ChannelRecord] = []
                observed_at = datetime.now(timezone.utc)
                for identifier in batch:
                    if identifier not in by_id:
                        report.unresolved[identifier] = "Channel details unavailable."
                    elif identifier in duplicate_ids:
                        report.unresolved[identifier] = "Duplicate channel details in response."
                    else:
                        try:
                            records.append(_channel_record(by_id[identifier], query, observed_at))
                        except (ValidationError, ValueError, KeyError, TypeError):
                            report.unresolved[identifier] = "Channel details failed validation."
                committed: list[tuple[ChannelRecord, SizeEligibilityResult]] = []
                with open_database(database_path) as connection:
                    for channel in records:
                        existing = get_channel(connection, channel.channel_id)
                        if existing is not None:
                            channel.discovery_query = existing.discovery_query
                        result = evaluate_size(channel, settings)
                        if save_channel(connection, channel):
                            save_size_result(connection, result)
                            committed.append((channel, result))
                        else:
                            report.unresolved[channel.channel_id] = "An observation newer than this run is already saved."
                for channel, result in committed:
                    report.channels.append(channel)
                    report.size_results.append(result)
                    report.saved_records += 1
    except YouTubeRequestError as error:
        report.error = str(error)

    if report.error:
        report.stopped_because = "API request failed; earlier completed batches remain saved."
    elif report.size_pass_count >= settings.discovery_target:
        report.stopped_because = "Size-eligible discovery target reached."
    elif report.search_calls >= settings.max_search_requests:
        report.stopped_because = "Configured search-request limit reached."
    else:
        report.stopped_because = "Available search pages exhausted."
    return report


def print_report(report: DiscoveryReport, settings: Settings) -> None:
    print("\nDiscovery summary")
    print(f"Channels collected this run (all sizes): {len(report.channels)}")
    print(f"Size-eligible channels this run: {report.size_pass_count}/{settings.discovery_target}")
    for status in ("PASS", "FAIL", "NEEDS_REVIEW"):
        count = sum(result.status == status for result in report.size_results)
        print(f"Size {status}: {count}")
    print(f"Channel IDs encountered in search pages: {report.candidate_ids_seen}")
    print(f"Records saved/updated this run: {report.saved_records}")
    print(f"Search calls: {report.search_calls}/{settings.max_search_requests}")
    print(f"Channel-detail calls: {report.channel_calls}")
    print(f"Unresolved channel IDs: {len(report.unresolved)}")
    print(f"Malformed search items skipped: {report.invalid_search_items}")
    print(report.stopped_because)
    if report.size_pass_count < settings.discovery_target:
        shortfall = settings.discovery_target - report.size_pass_count
        print(f"Size-eligible target not reached; {shortfall} more passing channels are needed.")
        print("Search limits and subscriber-size rules have not been changed.")
    if report.error:
        print(f"Request error: {report.error}")
    for channel, result in zip(report.channels, report.size_results):
        print(f"\n{json.dumps(channel.title, ensure_ascii=True)} | {channel.channel_id}")
        print(f"{result.status}: {result.reasons[0]}")
        print(str(channel.profile_url))
    for identifier, reason in report.unresolved.items():
        print(f"Unresolved {identifier}: {reason}")
    print("\nThese are size decisions only; campaign relevance is not evaluated yet.")


def self_check() -> None:
    """Exercise real discovery code through HTTPX's offline MockTransport."""
    ids = {letter: "UC" + letter * 22 for letter in "ABCDEFZ"}
    counts = {"A": "5000", "B": "100001", "C": "4999", "D": None, "E": "100000", "F": "8000"}
    # Two queries make the synthetic rotation reach page two on request three.
    settings = load_settings({"YOUTUBE_API_KEY": "synthetic-offline-key", "DISCOVERY_TARGET": "2",
                              "YOUTUBE_SEARCH_QUERIES": "synthetic query one|synthetic query two"})
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            if request.url.params.get("pageToken"):
                letters, token = "EF", None
            elif request.url.params["q"] == settings.search_queries[0]:
                letters, token = "AAZB", "synthetic-page-two"
            else:
                letters, token = "CD", None
            body: dict[str, Any] = {
                "items": [{"snippet": {"channelId": ids[letter]}} for letter in letters]
            }
            if token:
                body["nextPageToken"] = token
            return httpx.Response(200, json=body)
        items = []
        for identifier in reversed(request.url.params["id"].split(",")):
            letter = identifier[-1]
            if letter == "Z":
                continue
            stats: dict[str, Any] = {"hiddenSubscriberCount": letter == "D"}
            if counts[letter] is not None:
                stats["subscriberCount"] = counts[letter]
            items.append({
                "id": identifier, "snippet": {"title": f"Synthetic channel {letter}"},
                "statistics": stats,
                "contentDetails": {"relatedPlaylists": {"uploads": "UU" + letter * 22}},
            })
        return httpx.Response(200, json={"items": items})

    with TemporaryDirectory(prefix="outreach-discovery-check-") as directory:
        path = Path(directory) / "test.db"
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            first = discover(settings, client, path)
            second = discover(settings, client, path)
            limited = discover(replace(settings, max_search_requests=1), client, path)
        if first.error or len(first.channels) != 6 or ids["Z"] not in first.unresolved:
            raise RuntimeError("Offline discovery/omitted-channel check failed.")
        if [result.status for result in first.size_results] != ["PASS", "FAIL", "FAIL", "NEEDS_REVIEW", "PASS", "PASS"]:
            raise RuntimeError("Size thresholds or response-ID mapping failed.")
        if first.search_calls != 3 or len(second.channels) != 6 or limited.search_calls != 1:
            raise RuntimeError("Pagination, rerun, or search-limit check failed.")
        if len(limited.channels) != 2 or limited.size_pass_count != 1:
            raise RuntimeError("A shortfall was not preserved correctly.")
        with open_database(path) as connection:
            if record_counts(connection) != {"channels": 6, "videos": 0}:
                raise RuntimeError("Rerun created duplicate rows.")
        if first.size_pass_count != 3 or "target reached" not in first.stopped_because:
            raise RuntimeError("Size-eligible stopping rule or complete-batch saving failed.")

        def large_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/search"):
                start = 0 if request.url.params["q"] == settings.search_queries[0] else 50
                return httpx.Response(200, json={"items": [
                    {"snippet": {"channelId": "UC" + f"{number:022d}"}}
                    for number in range(start, start + 50)
                ]})
            return httpx.Response(200, json={"items": [
                {"id": identifier, "snippet": {"title": "Synthetic size-target check"},
                 "statistics": {"subscriberCount": "100001" if int(identifier[2:]) < 50 else "5000"}}
                for identifier in request.url.params["id"].split(",")
            ]})
        goal_path = Path(directory) / "size-goal.db"
        with httpx.Client(transport=httpx.MockTransport(large_handler)) as client:
            goal_report = discover(replace(settings, discovery_target=50), client, goal_path)
        if len(goal_report.channels) != 100 or goal_report.size_pass_count != 50 or goal_report.search_calls != 2:
            raise RuntimeError("Discovery stopped after 50 found instead of 50 size passes.")
        with open_database(goal_path) as connection:
            if record_counts(connection) != {"channels": 100, "videos": 0}:
                raise RuntimeError("Passing or rejected channels were not saved.")
    print("Discovery checks OK: size-eligible stopping rule, rejected-record saving, deduplication, pagination, request limits, and reruns.")
    print("Synthetic responses only; no network requests or project database changes.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Offline synthetic verification only")
    arguments = parser.parse_args()
    try:
        if arguments.self_check:
            self_check()
            return 0
        settings = load_settings()
        print(f"Discovery target: at least {settings.discovery_target} size-eligible channels this run.")
        print(f"Maximum search calls: {settings.max_search_requests}. No automatic retries.")
        with httpx.Client(
            headers={"X-Goog-Api-Key": settings.youtube_api_key},
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
        ) as client:
            report = discover(settings, client, settings.database_path)
        print_report(report, settings)
        with open_database(settings.database_path) as connection:
            counts = record_counts(connection)
            saved_passes = connection.execute(
                "SELECT COUNT(*) FROM influencers WHERE size_status = 'PASS'"
            ).fetchone()[0]
        print(f"Database: {settings.database_path}")
        print(f"Total saved channels: {counts['channels']}; total saved videos: {counts['videos']}")
        print(f"Total saved size-eligible channels: {saved_passes}")
        return 1 if report.error else (0 if report.size_pass_count >= settings.discovery_target else 2)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    except (sqlite3.Error, OSError):
        print("Database or file access failed. Check folder permissions and database availability.", file=sys.stderr)
        return 1
    except RuntimeError:
        print("Storage initialization or offline verification failed; existing data was not reset.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

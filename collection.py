"""Collect latest public, non-live uploads for saved size-eligible channels.

Default: five channels. --self-check uses synthetic HTTP responses and a
temporary database only. Metadata is current; samples store IDs, not snapshots.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx
from pydantic import ValidationError

from config import ConfigurationError, Settings, load_settings
from database import (
    get_collection_videos, get_size_result, list_channels,
    list_collection_results, open_database, record_counts,
    save_channel, save_collection_result, save_size_result, save_video,
)
from discovery import YouTubeRequestError, _request
from schemas import (
    ChannelRecord, CollectionExclusion, CollectionResult,
    SizeEligibilityResult, VideoRecord,
)


VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


def now() -> datetime:
    return datetime.now(timezone.utc)


def choose_channels(connection, settings: Settings, limit: int) -> list[ChannelRecord]:
    """Repeatable ID order, using saved PASS decisions and current size bounds."""
    eligible = []
    for channel in list_channels(connection):
        decision = get_size_result(connection, channel.channel_id)
        if (decision is not None and decision.status == "PASS"
                and channel.hidden_subscriber_count is not True
                and channel.subscriber_count is not None
                and settings.min_subscribers <= channel.subscriber_count <= settings.max_subscribers):
            eligible.append(channel)
    return sorted(eligible, key=lambda channel: channel.channel_id)[:limit]


def parse_video(item: dict, observed: datetime) -> VideoRecord:
    snippet = item.get("snippet", {})
    details = item.get("contentDetails", {})
    statistics = item.get("statistics", {})
    status = item.get("status", {})
    if not all(isinstance(section, dict) for section in (snippet, details, statistics, status)):
        raise ValueError("Invalid video sections.")
    return VideoRecord(
        video_id=item["id"], channel_id=snippet.get("channelId"),
        title=snippet.get("title"), description=snippet.get("description"),
        video_url=f"https://www.youtube.com/watch?v={item['id']}",
        published_at=snippet.get("publishedAt"), duration=details.get("duration"),
        views=statistics.get("viewCount"), likes=statistics.get("likeCount"),
        comments=statistics.get("commentCount"), privacy_status=status.get("privacyStatus"),
        live_broadcast_content=snippet.get("liveBroadcastContent"),
        default_language=snippet.get("defaultLanguage"),
        default_audio_language=snippet.get("defaultAudioLanguage"),
        dimension=details.get("dimension"), definition=details.get("definition"),
        projection=details.get("projection"), collected_at=observed,
    )


def collect_channel(channel: ChannelRecord, settings: Settings, client: httpx.Client,
                    run_id: str) -> tuple[CollectionResult, list[VideoRecord], bool]:
    started = now()
    candidates: list[VideoRecord] = []
    exclusions: list[CollectionExclusion] = []
    reasons: list[str] = []
    seen: set[str] = set()
    tokens: set[str] = set()
    token = None
    pages = requests = 0
    uncertain = failed = False
    exhausted = False
    previous_date = None

    def exclude(identifier: str, code: str, reason: str, evidence=()) -> None:
        exclusions.append(CollectionExclusion(
            video_id=identifier, reason_code=code, reason=reason, evidence=list(evidence)))

    if not channel.uploads_playlist_id:
        return CollectionResult(
            run_id=run_id, channel_id=channel.channel_id, status="SKIPPED",
            requested_video_count=settings.recent_video_limit,
            reasons=["The saved channel has no uploads playlist ID; refresh discovery to obtain it."],
            started_at=started, finished_at=now()), [], False

    try:
        for _ in range(settings.max_playlist_pages_per_channel):
            parameters = {"part": "contentDetails", "playlistId": channel.uploads_playlist_id,
                          "maxResults": 50}
            if token:
                parameters["pageToken"] = token
            pages += 1
            page = _request(client, "playlistItems", parameters)
            identifiers = []
            for entry in page["items"]:
                details = entry.get("contentDetails") if isinstance(entry, dict) else None
                identifier = details.get("videoId") if isinstance(details, dict) else None
                if not isinstance(identifier, str) or not VIDEO_ID.fullmatch(identifier):
                    uncertain = True
                    reasons.append("A malformed playlist entry had no usable video ID.")
                    continue
                if identifier not in seen:
                    seen.add(identifier)
                    identifiers.append(identifier)
            for offset in range(0, len(identifiers), 50):
                batch = identifiers[offset:offset + 50]
                requests += 1
                response = _request(client, "videos", {
                    "part": "snippet,contentDetails,statistics,status,liveStreamingDetails",
                    "id": ",".join(batch),
                })
                observed = now()
                mapped = {}
                for entry in response["items"]:
                    identifier = entry.get("id") if isinstance(entry, dict) else None
                    if identifier in batch:
                        if identifier in mapped:
                            raise YouTubeRequestError("YouTube returned duplicate video details; selection is uncertain.")
                        mapped[identifier] = entry
                for identifier in batch:
                    item = mapped.get(identifier)
                    if item is None:
                        exclude(identifier, "UNAVAILABLE", "Video details were not returned; eligibility is unknown.")
                        uncertain = True
                        continue
                    snippet = item.get("snippet", {})
                    status = item.get("status", {})
                    if not isinstance(snippet, dict) or not isinstance(status, dict):
                        exclude(identifier, "INVALID_METADATA", "Video sections were malformed.")
                        uncertain = True
                        continue
                    live = snippet.get("liveBroadcastContent")
                    if live == "upcoming":
                        exclude(identifier, "UPCOMING", "Upcoming content is excluded.",
                                ["snippet.liveBroadcastContent=upcoming"])
                        continue
                    if live == "live" or "liveStreamingDetails" in item:
                        live_evidence = [f"snippet.liveBroadcastContent={live}"]
                        if "liveStreamingDetails" in item:
                            live_evidence.append("liveStreamingDetails is present")
                            streaming = item["liveStreamingDetails"]
                            if isinstance(streaming, dict):
                                for field in ("scheduledStartTime", "scheduledEndTime", "actualStartTime", "actualEndTime"):
                                    if field in streaming:
                                        live_evidence.append(f"liveStreamingDetails.{field}=" +
                                                             json.dumps(streaming[field], ensure_ascii=True))
                        exclude(identifier, "LIVE_CONTENT", "Current or archived live content is excluded.",
                                live_evidence)
                        continue
                    if status.get("privacyStatus") in ("private", "unlisted"):
                        exclude(identifier, "NOT_PUBLIC", "Only public uploads are included.",
                                [f"status.privacyStatus={status['privacyStatus']}"])
                        continue
                    if status.get("uploadStatus") in ("deleted", "failed", "rejected", "uploaded"):
                        exclude(identifier, "UNAVAILABLE", "The upload is not confirmed as available processed content.",
                                [f"status.uploadStatus={status['uploadStatus']}"])
                        uncertain = True
                        continue
                    try:
                        video = parse_video(item, observed)
                    except (ValidationError, ValueError, TypeError, KeyError):
                        exclude(identifier, "INVALID_METADATA", "Video metadata failed validation.")
                        uncertain = True
                        continue
                    if video.channel_id != channel.channel_id:
                        exclude(identifier, "WRONG_CHANNEL", "Video ownership does not match this channel.")
                        uncertain = True
                        continue
                    if video.published_at is not None and video.published_at > observed:
                        exclude(identifier, "UPCOMING", "The publication timestamp is in the future.",
                                [f"snippet.publishedAt={video.published_at.isoformat()}"])
                        continue
                    if (video.privacy_status != "public" or video.live_broadcast_content != "none"
                            or video.published_at is None):
                        exclude(identifier, "UNKNOWN_ELIGIBILITY",
                                "Public visibility, non-live status, or publication time is missing.")
                        uncertain = True
                        continue
                    if previous_date is not None and video.published_at > previous_date:
                        uncertain = True
                        reasons.append("Playlist publication dates were not newest first; selection may be incomplete.")
                    previous_date = video.published_at
                    candidates.append(video)
            next_token = page.get("nextPageToken")
            if not next_token:
                exhausted = True
                break
            if not isinstance(next_token, str) or next_token in tokens:
                uncertain = True
                reasons.append("Pagination returned an invalid or repeated token; scanning stopped.")
                break
            tokens.add(next_token)
            token = next_token
            if len(candidates) >= settings.recent_video_limit and not uncertain:
                break
    except YouTubeRequestError as error:
        failed = uncertain = True
        reasons.append(str(error))

    candidates.sort(key=lambda video: (video.published_at, video.video_id), reverse=True)
    selected = candidates[:settings.recent_video_limit]
    if len(selected) < settings.recent_video_limit:
        reasons.append(f"Only {len(selected)}/{settings.recent_video_limit} eligible uploads were found.")
        reasons.append("Uploads playlist exhausted." if exhausted else "Scanning stopped before the playlist was exhausted.")
    if uncertain:
        reasons.append("The selected sample has unresolved gaps; review the exclusions and reasons.")
    if len(selected) == settings.recent_video_limit and not uncertain:
        outcome = "COMPLETE"
    else:
        outcome = "FAILED" if failed and not selected else "PARTIAL"
    result = CollectionResult(
        run_id=run_id, channel_id=channel.channel_id, uploads_playlist_id=channel.uploads_playlist_id,
        requested_video_count=settings.recent_video_limit,
        selected_video_ids=[video.video_id for video in selected], status=outcome,
        selection_uncertain=uncertain, exclusions=exclusions, reasons=list(dict.fromkeys(reasons)),
        playlist_pages=pages, video_requests=requests, started_at=started, finished_at=now(),
    )
    return result, selected, failed


def store_result(path: Path, result: CollectionResult, videos: list[VideoRecord]) -> None:
    # One transaction: either the videos AND ordered sample are saved, or neither.
    with open_database(path) as connection:
        for video in videos:
            save_video(connection, video)
        save_collection_result(connection, result)


def self_check() -> None:
    """Exercise the real collector with MockTransport, never live HTTP."""
    settings = load_settings({"YOUTUBE_API_KEY": "synthetic-key"})
    channel = ChannelRecord(
        channel_id="UC" + "A" * 22, title="Synthetic collector check",
        profile_url="https://www.youtube.com/channel/" + "UC" + "A" * 22,
        subscriber_count=10000, hidden_subscriber_count=False,
        uploads_playlist_id="UU" + "A" * 22, discovery_query="synthetic", collected_at=now(),
    )
    identifiers = [f"v{number:010d}" for number in range(20)]
    anchor = now() - timedelta(days=1)

    def detail(number):
        return {"id": identifiers[number], "snippet": {
            "channelId": channel.channel_id, "title": f"Synthetic {number}",
            "publishedAt": (anchor - timedelta(days=number)).isoformat(),
            "liveBroadcastContent": "none"},
            "contentDetails": {"duration": "PT30S" if number % 2 else "PT10M"},
            "statistics": {"viewCount": "0"}, "status": {"privacyStatus": "public"}}

    mode = "normal"
    def respond(request):
        nonlocal mode
        if mode == "error":
            return httpx.Response(403, json={"error": {"errors": [{"reason": "quotaExceeded"}]}})
        if request.url.path.endswith("playlistItems"):
            second = "pageToken" in request.url.params
            if second and mode == "late-error":
                return httpx.Response(503, json={"error": {}})
            numbers = list(range(8, 20)) if second else list(range(8))
            if mode in ("short", "late-error"):
                numbers = [8, 9]
            if mode == "gap":
                numbers = [4] + list(range(8, 20))
            payload = {"items": [{"contentDetails": {"videoId": identifiers[n]}} for n in numbers]}
            if not second and mode in ("normal", "late-error"):
                payload["nextPageToken"] = "second"
            if second and mode == "normal":
                payload["items"].append({"contentDetails": {"videoId": identifiers[8]}})
            return httpx.Response(200, json=payload)
        items = []
        for identifier in request.url.params["id"].split(","):
            number = identifiers.index(identifier)
            item = detail(number)
            if number == 0:
                item["snippet"]["liveBroadcastContent"] = "live"
            elif number == 1:
                item["snippet"]["liveBroadcastContent"] = "upcoming"
            elif number == 2:
                item["liveStreamingDetails"] = {"actualStartTime": anchor.isoformat()}
            elif number == 3:
                item["status"]["privacyStatus"] = "unlisted"
            elif number == 4:
                continue
            elif number == 5:
                item["snippet"]["publishedAt"] = (now() + timedelta(days=1)).isoformat()
            elif number == 6:
                item["snippet"]["channelId"] = "wrong-channel"
            elif number == 7:
                item["statistics"]["viewCount"] = -1
            items.append(item)
        return httpx.Response(200, json={"items": list(reversed(items))})

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result, videos, error = collect_channel(channel, settings, client, "check-1")
        require(not error and result.status == "PARTIAL" and result.selection_uncertain,
                "Unresolved metadata was incorrectly called complete.")
        require(result.selected_video_ids == identifiers[8:18], "ID mapping or newest-first selection failed.")
        require(len(result.exclusions) == 8 and result.playlist_pages == 2, "Exclusions or pagination failed.")
        require(all(v.views == 0 and v.likes is None for v in videos), "Unknown/zero counts changed.")
        with TemporaryDirectory(prefix="outreach-collector-check-") as directory:
            path = Path(directory) / "test.db"
            with open_database(path) as connection:
                save_channel(connection, channel)
                save_size_result(connection, SizeEligibilityResult(
                    channel_id=channel.channel_id, status="PASS", reasons=["Synthetic size check"], evaluated_at=now()))
                require(choose_channels(connection, settings, 5) == [channel], "Channel selection failed.")
            store_result(path, result, videos)
            mode = "gap"
            repeated, refreshed, _ = collect_channel(channel, settings, client, "check-2")
            store_result(path, repeated, refreshed)
            with open_database(path) as connection:
                require(record_counts(connection)["videos"] == 10, "Rerun duplicated videos.")
                require(len(list_collection_results(connection, channel.channel_id)) == 2, "Sample history lost.")
                require([v.video_id for v in get_collection_videos(connection, "check-1", channel.channel_id)]
                        == result.selected_video_ids, "Sample membership changed.")
                require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "Storage integrity failed.")
        mode = "normal"
        capped, _, _ = collect_channel(channel, replace(settings, max_playlist_pages_per_channel=1), client, "cap")
        require(capped.status == "PARTIAL" and capped.playlist_pages == 1, "Page cap ignored.")
        mode = "short"
        short, _, _ = collect_channel(channel, settings, client, "short")
        require(short.status == "PARTIAL" and len(short.selected_video_ids) == 2, "Shortfall not reported.")
        mode = "late-error"
        partial, retained, stopped = collect_channel(channel, settings, client, "late-error")
        require(stopped and partial.status == "PARTIAL" and len(retained) == 2,
                "A later API failure discarded previously collected videos.")
        mode = "error"
        failed, _, stopped = collect_channel(channel, settings, client, "failed")
        require(stopped and failed.status == "FAILED" and failed.playlist_pages == 1, "API failure/retry handling failed.")
        mode = "gap"
        # Remove the unavailable entry to exercise an unambiguous complete sample.
        def complete_response(request):
            response = respond(request)
            body = response.json()
            if request.url.path.endswith("playlistItems"):
                body["items"] = body["items"][1:]
            return httpx.Response(200, json=body)
        with httpx.Client(transport=httpx.MockTransport(complete_response)) as clean_client:
            complete, _, _ = collect_channel(channel, settings, clean_client, "complete")
            require(complete.status == "COMPLETE", "Valid complete sample was rejected.")
        skipped, _, _ = collect_channel(channel.model_copy(update={"uploads_playlist_id": None}), settings, client, "skip")
        require(skipped.status == "SKIPPED" and skipped.playlist_pages == 0, "Missing playlist not skipped.")
    print("Collection checks OK: latest samples, live/upcoming exclusions, missing data, pagination, page limits, errors, persistence, and reruns.")
    print("Synthetic responses only; no network requests or project database changes.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--limit", type=int, default=5, help="Number of saved size passes to attempt (default: 5).")
    args = parser.parse_args()
    if not 1 <= args.limit <= 5000:
        parser.error("--limit must be between 1 and 5000.")
    if args.self_check:
        self_check()
        return 0
    try:
        settings = load_settings()
        if not settings.database_path.is_file():
            print("Database missing. Run database.py and discovery.py first.", file=sys.stderr)
            return 1
        with open_database(settings.database_path) as connection:
            channels = choose_channels(connection, settings, args.limit)
        if not channels:
            print("No saved size-eligible channels are available; run discovery first.")
            return 1
        run_id = uuid4().hex
        print(f"Collection run: {run_id}")
        print(f"Selected channels: {len(channels)}/{args.limit}, ordered by channel ID.")
        print(f"Up to {settings.recent_video_limit} latest public non-live uploads each; regular videos and Shorts share the sample.")
        print("Saved subscriber decisions are used; subscriber counts are not refreshed here.")
        print("Video format remains unconfirmed; duration alone is never used to label Shorts.")
        print("No automatic retries. Stop after an API error; earlier committed channels remain saved.")
        outcomes = []
        with httpx.Client(headers={"X-Goog-Api-Key": settings.youtube_api_key},
                          timeout=settings.request_timeout_seconds, follow_redirects=False) as client:
            for channel in channels:
                result, videos, stop = collect_channel(channel, settings, client, run_id)
                store_result(settings.database_path, result, videos)
                outcomes.append(result)
                print(f"\n{json.dumps(channel.title, ensure_ascii=True)} | {channel.channel_id}")
                print(f"{result.status}: {len(videos)}/{result.requested_video_count} videos; excluded {len(result.exclusions)}; playlist calls {result.playlist_pages}; video-detail calls {result.video_requests}.")
                for reason in result.reasons:
                    print(f"Reason: {reason}")
                for video in videos:
                    print(f"{video.published_at.isoformat()} | {json.dumps(video.title, ensure_ascii=True)} | {video.video_url}")
                if stop:
                    print("API error: remaining selected channels were not attempted.")
                    break
        with open_database(settings.database_path) as connection:
            counts = record_counts(connection)
        print(f"\nChannels attempted: {len(outcomes)}/{args.limit}; complete samples: {sum(r.status == 'COMPLETE' for r in outcomes)}.")
        print(f"Sample videos saved/updated this run: {sum(len(r.selected_video_ids) for r in outcomes)}")
        print(f"Total saved channels: {counts['channels']}; total saved videos: {counts['videos']}")
        print(f"Database: {settings.database_path}")
        print("Sample IDs and outcomes were saved; no per-run video snapshots. Campaign relevance is not evaluated yet.")
        return 0 if len(outcomes) == args.limit and all(r.status == "COMPLETE" for r in outcomes) else 2
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    except (sqlite3.Error, ValueError, OSError):
        print("Collection/storage validation failed. The current channel transaction was rolled back; inspect the files and database before rerunning.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

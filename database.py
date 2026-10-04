"""SQLite persistence for collection, classification, filtering and enrichment evidence.

No API calls or credentials.

Use `with open_database(path) as connection:` to group writes atomically.
Run this file to verify storage and initialize data/outreach.db.
"""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import json
import hashlib
from tempfile import TemporaryDirectory
from typing import Iterator

from config import DEFAULT_DATABASE_PATH, database_path, load_environment
from schemas import (
    ChannelRecord,
    CollectionResult,
    SizeEligibilityResult,
    VideoRecord,
    ClassificationRecord,
    ChannelFilteringResult,
    TitleClassificationRecord,
    TitleChannelFilteringResult,
)


ClassificationStorageRecord = ClassificationRecord | TitleClassificationRecord
FilteringStorageRecord = ChannelFilteringResult | TitleChannelFilteringResult


def _parse_classification_record(value) -> ClassificationStorageRecord:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        raise ValueError("Classification record must be an object.")
    if "schema_version" in data:
        return TitleClassificationRecord.model_validate(data)
    if data.get("criteria_version") == "python_course_v2":
        raise ValueError("The new criteria require the versioned title-only record format.")
    return ClassificationRecord.model_validate(data)


def _parse_filtering_record(value) -> FilteringStorageRecord:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        raise ValueError("Filtering record must be an object.")
    if "schema_version" in data:
        return TitleChannelFilteringResult.model_validate(data)
    if isinstance(data.get("rules"), dict) and data["rules"].get("version") == "python_course_v2":
        raise ValueError("The new rules require the versioned title-only filtering format.")
    return ChannelFilteringResult.model_validate(data)


# Existing classification/filtering formats remain readable.
# Version 4 adds enrichment evidence tables without rewriting historical data.
SCHEMA_VERSION = 4


def _initialize(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    if version not in (0, 1, 2, 3):
        raise RuntimeError(f"Unsupported database schema version: {version}.")

    # Explicit BEGIN makes both DDL and the version update atomic.
    connection.execute("BEGIN IMMEDIATE")
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            connection.commit()
            return

        if version == 0:
            _create_initial_tables(connection)
        elif version == 1:
            connection.execute(
                "SELECT channel_id, record_json, size_status, "
                "size_result_json FROM influencers LIMIT 0"
            )
            connection.execute(
                "SELECT video_id, channel_id, record_json FROM videos LIMIT 0"
            )
        elif version in (2, 3):
            connection.execute(
                "SELECT run_id, channel_id, record_json "
                "FROM collection_results LIMIT 0"
            )
            connection.execute(
                "SELECT run_id, channel_id, position, video_id "
                "FROM collection_sample_videos LIMIT 0"
            )
        else:
            raise RuntimeError(
                f"Unsupported database schema version: {version}."
            )

        if version < 2:
            _create_collection_tables(connection)

        if version < 3:
            _create_filtering_tables(connection)
        _create_enrichment_evidence_tables(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _create_enrichment_evidence_tables(connection: sqlite3.Connection) -> None:
    # These names do not overlap with enrichment.py's existing profile/cache tables.
    connection.execute("""CREATE TABLE IF NOT EXISTS enrichment_sources (
        evidence_id TEXT PRIMARY KEY,
        channel_id TEXT NOT NULL REFERENCES influencers(channel_id),
        record_json TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS enrichment_batches (
        batch_id TEXT PRIMARY KEY,
        channel_id TEXT NOT NULL REFERENCES influencers(channel_id),
        record_json TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS enrichment_batch_sources (
        batch_id TEXT NOT NULL REFERENCES enrichment_batches(batch_id),
        evidence_id TEXT NOT NULL REFERENCES enrichment_sources(evidence_id),
        position INTEGER NOT NULL CHECK(position >= 0),
        PRIMARY KEY(batch_id, evidence_id), UNIQUE(batch_id, position)
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS enrichment_sources_channel ON enrichment_sources(channel_id)")


def _evidence_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _evidence_time(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("Evidence timestamps must be ISO strings with a timezone.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Evidence timestamps must include a timezone.")


def save_enrichment_source(connection: sqlite3.Connection, source: dict) -> str:
    """Save immutable fetched/saved-source evidence; return its stable ID.

    Required keys: channel_id, source_id, kind, url (string or None), text,
    observed_at (aware ISO string), status, reason (string or None).
    BLOCKED/ERROR/LIMIT_REACHED distinguish unfinished work from an empty page.
    This helper never fetches data or judges contact ownership.
    """
    required = {"channel_id", "source_id", "kind", "url", "text", "observed_at", "status", "reason"}
    if not isinstance(source, dict) or set(source) != required:
        raise ValueError("Unexpected enrichment source fields.")
    for key in ("channel_id", "source_id", "kind", "status"):
        if not isinstance(source[key], str) or not source[key].strip():
            raise ValueError("Source identifiers and labels must be nonempty strings.")
    if source["kind"] not in ("CHANNEL_DESCRIPTION", "VIDEO_DESCRIPTION", "WEBSITE", "INSTAGRAM", "PUBLIC_SOURCE"):
        raise ValueError("Unknown enrichment source kind.")
    if source["status"] not in ("AVAILABLE", "BLOCKED", "ERROR", "LIMIT_REACHED"):
        raise ValueError("Unknown enrichment source status.")
    if not isinstance(source["text"], str):
        raise ValueError("Source text must be a string.")
    for key in ("url", "reason"):
        if source[key] is not None and not isinstance(source[key], str):
            raise ValueError("Source URL/reason must be a string or None.")
    if source["status"] != "AVAILABLE" and not source["reason"]:
        raise ValueError("Unfinished evidence needs a reason.")
    _evidence_time(source["observed_at"])
    payload = _evidence_json(source)
    identifier = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    connection.execute("INSERT OR IGNORE INTO enrichment_sources VALUES (?, ?, ?)",
                       (identifier, source["channel_id"], payload))
    return identifier


def get_enrichment_source(connection: sqlite3.Connection, evidence_id: str) -> dict | None:
    row = connection.execute("SELECT record_json FROM enrichment_sources WHERE evidence_id = ?",
                             (evidence_id,)).fetchone()
    return json.loads(row[0]) if row else None


def save_enrichment_batch(connection: sqlite3.Connection, channel_id: str,
                          evidence_ids: list[str], candidates: list[dict], context: dict) -> str:
    """Store a reusable candidate/evidence batch before an LLM call.

    Each candidate has kind EMAIL/WEBSITE/INSTAGRAM, a literal value, and
    evidence_ids. Context contains input/pipeline versions and relevant model
    settings. Ownership judgments belong to later validated LLM results.
    Caller deduplicates candidates and merges all their evidence references.
    """
    if not isinstance(evidence_ids, list) or not evidence_ids or any(not isinstance(i, str) for i in evidence_ids):
        raise ValueError("A batch needs source evidence IDs.")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("Batch evidence IDs must be unique.")
    if not isinstance(context, dict) or not context or not isinstance(candidates, list):
        raise ValueError("A batch needs version/context settings and a candidate list.")
    sources = {identifier: get_enrichment_source(connection, identifier) for identifier in evidence_ids}
    if any(source is None or source["channel_id"] != channel_id for source in sources.values()):
        raise ValueError("Batch evidence must exist and belong to the same channel.")
    seen = set()
    for item in candidates:
        if not isinstance(item, dict) or set(item) != {"kind", "value", "evidence_ids"}:
            raise ValueError("Unexpected candidate fields.")
        kind, value, references = item["kind"], item["value"], item["evidence_ids"]
        if kind not in ("EMAIL", "WEBSITE", "INSTAGRAM") or not isinstance(value, str) or not value.strip():
            raise ValueError("Invalid candidate type/value.")
        if not isinstance(references, list) or not references or any(not isinstance(i, str) for i in references):
            raise ValueError("Candidates require evidence references.")
        if len(set(references)) != len(references) or not set(references).issubset(sources):
            raise ValueError("Candidate evidence must be unique and included in this batch.")
        if (kind, value) in seen:
            raise ValueError("Merge duplicate candidates before saving the batch.")
        seen.add((kind, value))
        for identifier in references:
            source = sources[identifier]
            # Normalization is for matching only; original fetched text is retained.
            if source["status"] != "AVAILABLE" or value.casefold() not in source["text"].casefold():
                raise ValueError("Candidate must be present in each cited available source.")
    record = {"channel_id": channel_id, "evidence_ids": evidence_ids, "candidates": candidates, "context": context}
    payload = _evidence_json(record)
    batch_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # An outer transaction prevents RELEASE from committing the caller's work.
    # Like the other save helpers, the caller owns commit/rollback.
    if not connection.in_transaction:
        connection.execute("BEGIN")
    connection.execute("SAVEPOINT enrichment_batch_write")
    try:
        connection.execute("INSERT OR IGNORE INTO enrichment_batches VALUES (?, ?, ?)",
                           (batch_id, channel_id, payload))
        for position, identifier in enumerate(evidence_ids):
            connection.execute("INSERT OR IGNORE INTO enrichment_batch_sources VALUES (?, ?, ?)",
                               (batch_id, identifier, position))
        connection.execute("RELEASE SAVEPOINT enrichment_batch_write")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT enrichment_batch_write")
        connection.execute("RELEASE SAVEPOINT enrichment_batch_write")
        raise
    return batch_id


def get_enrichment_batch(connection: sqlite3.Connection, batch_id: str) -> dict | None:
    row = connection.execute("SELECT record_json FROM enrichment_batches WHERE batch_id = ?", (batch_id,)).fetchone()
    return json.loads(row[0]) if row else None


def _create_collection_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE UNIQUE INDEX videos_identity ON videos(video_id, channel_id)"
    )
    connection.execute(
        """CREATE TABLE collection_results (
            run_id TEXT NOT NULL,
            channel_id TEXT NOT NULL REFERENCES influencers(channel_id),
            status TEXT NOT NULL CHECK(
                status IN ('COMPLETE', 'PARTIAL', 'FAILED', 'SKIPPED')
            ),
            finished_at TEXT NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY(run_id, channel_id)
        )"""
    )
    connection.execute(
        """CREATE TABLE collection_sample_videos (
            run_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            position INTEGER NOT NULL CHECK(position >= 0),
            video_id TEXT NOT NULL,
            PRIMARY KEY(run_id, channel_id, position),
            UNIQUE(run_id, channel_id, video_id),
            FOREIGN KEY(run_id, channel_id)
                REFERENCES collection_results(run_id, channel_id),
            FOREIGN KEY(video_id, channel_id)
                REFERENCES videos(video_id, channel_id)
        )"""
    )
    connection.execute(
        "CREATE INDEX collection_results_channel_time "
        "ON collection_results(channel_id, finished_at)"
    )


def _create_filtering_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE classifications (
            classification_id TEXT PRIMARY KEY NOT NULL,
            collection_run_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            input_fingerprint TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            criteria_version TEXT NOT NULL,
            classified_at TEXT NOT NULL,
            record_json TEXT NOT NULL,
            UNIQUE(classification_id, channel_id),
            UNIQUE(
                channel_id, input_fingerprint, model,
                prompt_version, criteria_version
            ),
            FOREIGN KEY(collection_run_id, channel_id)
                REFERENCES collection_results(run_id, channel_id)
        )"""
    )
    connection.execute(
        """CREATE TABLE filtering_results (
            filtering_id TEXT PRIMARY KEY NOT NULL,
            collection_run_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            classification_id TEXT,
            input_fingerprint TEXT NOT NULL,
            rules_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK(
                status IN ('PASS', 'FAIL', 'NEEDS_REVIEW')
            ),
            evaluated_at TEXT NOT NULL,
            record_json TEXT NOT NULL,
            UNIQUE(
                collection_run_id, channel_id,
                input_fingerprint, rules_version
            ),
            FOREIGN KEY(collection_run_id, channel_id)
                REFERENCES collection_results(run_id, channel_id),
            FOREIGN KEY(classification_id, channel_id)
                REFERENCES classifications(classification_id, channel_id)
        )"""
    )
    connection.execute(
        "CREATE INDEX filtering_results_cohort "
        "ON filtering_results(collection_run_id, channel_id, evaluated_at)"
    )


def _create_initial_tables(connection: sqlite3.Connection) -> None:
    existing = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    if existing:
        raise RuntimeError(
            "Existing unversioned database requires explicit migration."
        )

    # The caller owns the schema transaction.
    connection.execute(
        """CREATE TABLE influencers (
            channel_id TEXT PRIMARY KEY NOT NULL,
            record_json TEXT NOT NULL,
            size_status TEXT CHECK(
                size_status IN ('PASS', 'FAIL', 'NEEDS_REVIEW')
            ),
            size_result_json TEXT,
            CHECK (
                (size_status IS NULL AND size_result_json IS NULL)
                OR (
                    size_status IS NOT NULL
                    AND size_result_json IS NOT NULL
                )
            )
        )"""
    )
    connection.execute(
        """CREATE TABLE videos (
            video_id TEXT PRIMARY KEY NOT NULL,
            channel_id TEXT NOT NULL REFERENCES influencers(channel_id),
            record_json TEXT NOT NULL
        )"""
    )
    connection.execute(
        "CREATE INDEX videos_channel_id ON videos(channel_id)"
    )


def _backup_before_migration(
    connection: sqlite3.Connection, path: Path
) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version not in (1, 2, 3):
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = path.with_name(
        f"{path.stem}.v{version}-backup-{stamp}{path.suffix}"
    )

    # Exclusive creation prevents an existing backup from being overwritten.
    with backup_path.open("xb"):
        pass

    backup = sqlite3.connect(backup_path)
    try:
        connection.backup(backup)
    finally:
        backup.close()

    print(f"Pre-migration backup: {backup_path}")


@contextmanager
def open_database(path: Path) -> Iterator[sqlite3.Connection]:
    """Initialize safely, commit on success, roll back on error, always close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        _backup_before_migration(connection, path)
        _initialize(connection)
        with connection:
            yield connection
    finally:
        connection.close()


def get_channel(
    connection: sqlite3.Connection, channel_id: str
) -> ChannelRecord | None:
    row = connection.execute(
        "SELECT record_json FROM influencers WHERE channel_id = ?",
        (channel_id,),
    ).fetchone()
    return (
        None
        if row is None
        else ChannelRecord.model_validate_json(row["record_json"])
    )


def save_channel(
    connection: sqlite3.Connection, channel: ChannelRecord
) -> bool:
    """Insert or refresh one channel; ignore observations older than saved data.

    A changed observation clears its previous size decision to avoid stale results.
    An identical rerun keeps the decision. No commit occurs inside this function.
    """
    channel = ChannelRecord.model_validate(channel.model_dump())
    existing = get_channel(connection, channel.channel_id)
    if existing is not None and channel.collected_at < existing.collected_at:
        return False

    connection.execute(
        """INSERT INTO influencers(channel_id, record_json) VALUES (?, ?)
        ON CONFLICT(channel_id) DO UPDATE SET
            size_status = CASE
                WHEN influencers.record_json = excluded.record_json
                THEN influencers.size_status ELSE NULL END,
            size_result_json = CASE
                WHEN influencers.record_json = excluded.record_json
                THEN influencers.size_result_json ELSE NULL END,
            record_json = excluded.record_json""",
        (channel.channel_id, channel.model_dump_json()),
    )
    return True


def save_size_result(
    connection: sqlite3.Connection, result: SizeEligibilityResult
) -> None:
    result = SizeEligibilityResult.model_validate(result.model_dump())
    channel = get_channel(connection, result.channel_id)
    if channel is None:
        raise ValueError("Save the channel before its size eligibility result.")
    if result.evaluated_at < channel.collected_at:
        raise ValueError(
            "Size evaluation cannot predate the current observation."
        )

    previous = get_size_result(connection, result.channel_id)
    if previous is not None and result.evaluated_at < previous.evaluated_at:
        raise ValueError(
            "An older size evaluation cannot replace a newer one."
        )

    connection.execute(
        "UPDATE influencers SET size_status = ?, size_result_json = ? "
        "WHERE channel_id = ?",
        (result.status, result.model_dump_json(), result.channel_id),
    )


def get_size_result(
    connection: sqlite3.Connection, channel_id: str
) -> SizeEligibilityResult | None:
    row = connection.execute(
        "SELECT size_result_json FROM influencers WHERE channel_id = ?",
        (channel_id,),
    ).fetchone()
    if row is None or row["size_result_json"] is None:
        return None
    return SizeEligibilityResult.model_validate_json(row["size_result_json"])


def get_video(
    connection: sqlite3.Connection, video_id: str
) -> VideoRecord | None:
    row = connection.execute(
        "SELECT record_json FROM videos WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    return (
        None
        if row is None
        else VideoRecord.model_validate_json(row["record_json"])
    )


def save_video(
    connection: sqlite3.Connection, video: VideoRecord
) -> bool:
    """Upsert by video ID, preserving ownership and ignoring older observations."""
    video = VideoRecord.model_validate(video.model_dump())
    existing = get_video(connection, video.video_id)
    if existing is not None:
        if existing.channel_id != video.channel_id:
            raise ValueError(
                "An existing video cannot be reassigned to another channel."
            )
        if video.collected_at < existing.collected_at:
            return False

    connection.execute(
        """INSERT INTO videos(video_id, channel_id, record_json)
        VALUES (?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            record_json = excluded.record_json""",
        (video.video_id, video.channel_id, video.model_dump_json()),
    )
    return True


def list_channels(
    connection: sqlite3.Connection,
) -> list[ChannelRecord]:
    rows = connection.execute(
        "SELECT record_json FROM influencers ORDER BY channel_id"
    )
    return [
        ChannelRecord.model_validate_json(row["record_json"])
        for row in rows
    ]


def list_videos(
    connection: sqlite3.Connection, channel_id: str
) -> list[VideoRecord]:
    """Return stored videos by ID; these may include earlier observations."""
    rows = connection.execute(
        "SELECT record_json FROM videos "
        "WHERE channel_id = ? ORDER BY video_id",
        (channel_id,),
    )
    return [
        VideoRecord.model_validate_json(row["record_json"])
        for row in rows
    ]


def record_counts(
    connection: sqlite3.Connection,
) -> dict[str, int]:
    return {
        "channels": connection.execute(
            "SELECT COUNT(*) FROM influencers"
        ).fetchone()[0],
        "videos": connection.execute(
            "SELECT COUNT(*) FROM videos"
        ).fetchone()[0],
    }


def get_collection_result(
    connection: sqlite3.Connection, run_id: str, channel_id: str
) -> CollectionResult | None:
    row = connection.execute(
        "SELECT record_json FROM collection_results "
        "WHERE run_id = ? AND channel_id = ?",
        (run_id, channel_id),
    ).fetchone()
    return (
        None
        if row is None
        else CollectionResult.model_validate_json(row["record_json"])
    )


def save_collection_result(
    connection: sqlite3.Connection, result: CollectionResult
) -> bool:
    """Save an immutable result and ordered video-ID links.

    Save selected videos first. Repeating an identical result is harmless;
    changing a processed result requires a new run ID.
    """
    result = CollectionResult.model_validate(result.model_dump())
    existing = get_collection_result(
        connection, result.run_id, result.channel_id
    )
    if existing is not None:
        if existing != result:
            raise ValueError(
                "A saved collection result cannot be changed; use a new run ID."
            )
        return False

    if get_channel(connection, result.channel_id) is None:
        raise ValueError("Save the channel before its collection result.")

    previous_date: datetime | None = None
    for video_id in result.selected_video_ids:
        video = get_video(connection, video_id)
        if video is None or video.channel_id != result.channel_id:
            raise ValueError(
                "Selected videos must already be saved for this channel."
            )
        if video.privacy_status != "public":
            raise ValueError(
                "Selected videos must have confirmed public visibility."
            )
        if (
            video.live_broadcast_content != "none"
            or video.live_streaming_details is not None
        ):
            raise ValueError(
                "Live, upcoming, or uncertain live status cannot enter the sample."
            )
        if (
            video.published_at is None
            or video.published_at > result.finished_at
        ):
            raise ValueError(
                "Selected videos need an already-published timestamp."
            )
        if not (
            result.started_at <= video.collected_at <= result.finished_at
        ):
            raise ValueError(
                "Selected video metadata must be observed during this collection run."
            )
        if (
            previous_date is not None
            and video.published_at > previous_date
        ):
            raise ValueError(
                "Selected videos must be ordered newest first."
            )
        previous_date = video.published_at

    connection.execute(
        "INSERT INTO collection_results"
        "(run_id, channel_id, status, finished_at, record_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            result.run_id,
            result.channel_id,
            result.status,
            result.finished_at.astimezone(timezone.utc).isoformat(),
            result.model_dump_json(),
        ),
    )
    for position, video_id in enumerate(result.selected_video_ids):
        connection.execute(
            "INSERT INTO collection_sample_videos"
            "(run_id, channel_id, position, video_id) "
            "VALUES (?, ?, ?, ?)",
            (result.run_id, result.channel_id, position, video_id),
        )
    return True


def list_collection_results(
    connection: sqlite3.Connection, channel_id: str
) -> list[CollectionResult]:
    rows = connection.execute(
        "SELECT record_json FROM collection_results "
        "WHERE channel_id = ? ORDER BY finished_at DESC, run_id DESC",
        (channel_id,),
    )
    return [
        CollectionResult.model_validate_json(row["record_json"])
        for row in rows
    ]


def get_collection_videos(
    connection: sqlite3.Connection, run_id: str, channel_id: str
) -> list[VideoRecord]:
    """Read current video metadata for saved sample IDs, in sample order.

    Metadata may change on refresh; no per-run copy is kept.
    """
    rows = connection.execute(
        """SELECT videos.record_json
        FROM collection_sample_videos AS sample
        JOIN videos
            ON videos.video_id = sample.video_id
            AND videos.channel_id = sample.channel_id
        WHERE sample.run_id = ? AND sample.channel_id = ?
        ORDER BY sample.position""",
        (run_id, channel_id),
    )
    return [
        VideoRecord.model_validate_json(row["record_json"])
        for row in rows
    ]


def list_run_collection_results(
    connection: sqlite3.Connection, run_id: str
) -> list[CollectionResult]:
    """Select a named collection cohort."""
    rows = connection.execute(
        "SELECT record_json FROM collection_results "
        "WHERE run_id = ? ORDER BY channel_id",
        (run_id,),
    )
    return [
        CollectionResult.model_validate_json(row[0])
        for row in rows
    ]


def get_classification(
    connection: sqlite3.Connection, classification_id: str
) -> ClassificationStorageRecord | None:
    row = connection.execute(
        "SELECT record_json FROM classifications "
        "WHERE classification_id = ?",
        (classification_id,),
    ).fetchone()
    return (
        None
        if row is None
        else _parse_classification_record(row[0])
    )


def find_reusable_classification(
    connection: sqlite3.Connection,
    *,
    channel_id: str,
    input_fingerprint: str,
    model: str,
    prompt_version: str,
    criteria_version: str,
    sample_video_ids: list[str],
) -> ClassificationStorageRecord | None:
    """Caller hashes the exact supplied text and context, excluding counters.

    Matching run IDs alone never establish reuse. Saved judgments are immutable
    and can be reused across runs with the same ordered sample and input.
    """
    row = connection.execute(
        """SELECT record_json FROM classifications
        WHERE channel_id = ?
        AND input_fingerprint = ?
        AND model = ?
        AND prompt_version = ?
        AND criteria_version = ?""",
        (
            channel_id,
            input_fingerprint,
            model,
            prompt_version,
            criteria_version,
        ),
    ).fetchone()
    if row is None:
        return None

    record = _parse_classification_record(row[0])
    return (
        record
        if record.sample_video_ids == sample_video_ids
        else None
    )


def _linked_sample(
    connection, run_id, channel_id, identifiers, evaluated_at
):
    collection = get_collection_result(connection, run_id, channel_id)
    if (
        collection is None
        or collection.selected_video_ids != identifiers
    ):
        raise ValueError(
            "The record must reference an existing collection's exact ordered sample."
        )

    channel = get_channel(connection, channel_id)
    videos = get_collection_videos(connection, run_id, channel_id)
    if (
        channel is None
        or [video.video_id for video in videos] != identifiers
    ):
        raise ValueError(
            "Sample records are missing or belong to another channel."
        )

    latest_source_time = max(
        [collection.finished_at, channel.collected_at]
        + [video.collected_at for video in videos]
    )
    if evaluated_at < latest_source_time:
        raise ValueError(
            "Evaluation cannot predate its source observations or collection."
        )
    return collection, channel, videos


def save_classification(
    connection: sqlite3.Connection, record: ClassificationStorageRecord
) -> bool:
    """Validate evidence and persist once, without committing."""
    record = _parse_classification_record(record.model_dump())
    existing = get_classification(connection, record.classification_id)
    if existing is not None:
        if existing != record:
            raise ValueError(
                "Saved classifications are immutable; use a new input or version."
            )
        return False

    collection, channel, videos = _linked_sample(
        connection,
        record.collection_run_id,
        record.channel_id,
        record.sample_video_ids,
        record.classified_at,
    )
    if isinstance(record, TitleClassificationRecord) and (
        collection.status != "COMPLETE" or collection.selection_uncertain
        or collection.requested_video_count != 10
    ):
        raise ValueError("Title-only classification requires a complete, certain ten-video sample.")
    record.response.validate_sources(channel, videos)

    # Reuse lookup must precede a new LLM request in the future caller.
    connection.execute(
        "INSERT INTO classifications "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.classification_id,
            record.collection_run_id,
            record.channel_id,
            record.input_fingerprint,
            record.model,
            record.prompt_version,
            record.criteria_version,
            record.classified_at.astimezone(timezone.utc).isoformat(),
            record.model_dump_json(),
        ),
    )
    return True


def get_filtering_result(
    connection: sqlite3.Connection, filtering_id: str
) -> FilteringStorageRecord | None:
    row = connection.execute(
        "SELECT record_json FROM filtering_results "
        "WHERE filtering_id = ?",
        (filtering_id,),
    ).fetchone()
    return (
        None
        if row is None
        else _parse_filtering_record(row[0])
    )


def save_filtering_result(
    connection: sqlite3.Connection, result: FilteringStorageRecord
) -> bool:
    """Persist decisions, arithmetic inputs, reasons, and limitation.

    Current metadata must match the inputs at insertion. Later refreshes do not
    rewrite historical decisions. No per-run video metadata snapshots are added.
    """
    result = _parse_filtering_record(result.model_dump())
    existing = get_filtering_result(connection, result.filtering_id)
    if existing is not None:
        if existing != result:
            raise ValueError(
                "Saved filtering results are immutable; create a new evaluation."
            )
        return False

    collection, channel, videos = _linked_sample(
        connection,
        result.collection_run_id,
        result.channel_id,
        result.sample_video_ids,
        result.evaluated_at,
    )
    if (
        result.selection_uncertain != collection.selection_uncertain
        or result.rules.requested_video_count
        != collection.requested_video_count
    ):
        raise ValueError(
            "Filtering must preserve the collection's sample size and uncertainty."
        )

    title_only = isinstance(result, TitleChannelFilteringResult)
    if title_only:
        available = bool((channel.description or "").strip())
        if result.technology.description_available != available:
            raise ValueError("Niche description availability must match the saved channel.")

    if result.classification_id is None:
        relevance = result.content_relevance
        if (relevance.status != "NEEDS_REVIEW" or relevance.qualifying_video_ids
                or relevance.unclear_video_ids):
            raise ValueError("Without classification, title relevance must remain unresolved.")
        if title_only:
            if result.technology.status != "NEEDS_REVIEW" or result.technology.label is not None:
                raise ValueError("Without classification, channel niche must remain unresolved.")
        elif (result.technology.status != "NEEDS_REVIEW"
                or result.technology.qualifying_video_ids or result.technology.unclear_video_ids):
            raise ValueError("Without classification, technology decisions must remain unresolved.")
    else:
        classification = get_classification(connection, result.classification_id)
        if (
            classification is None
            or classification.channel_id != result.channel_id
            or classification.sample_video_ids != result.sample_video_ids
            or classification.criteria_version != result.rules.version
            or classification.classified_at > result.evaluated_at
            or isinstance(classification, TitleClassificationRecord) != title_only
        ):
            raise ValueError("Classification identity, sample, format, version, or timestamp does not match.")

        # Callers must fingerprint current supplied text before choosing reuse.
        # Cached judgments may originate in an earlier collection run.
        classification.response.validate_sources(channel, videos)
        labels = {item.video_id: item for item in classification.response.videos}
        if title_only:
            niche = classification.response.niche
            if (result.technology.label != niche.label
                    or result.technology.evidence_source != niche.evidence_source):
                raise ValueError("Niche decision must match the saved channel-description judgment.")
            checks = ((result.content_relevance, "relevance_label", ("MATCH", "RELATED")),)
        else:
            checks = (
                (result.technology, "technology_label", ("TECHNOLOGY",)),
                (result.content_relevance, "relevance_label", ("MATCH", "RELATED")),
            )
        for criterion, label_name, qualifying_labels in checks:
            matches = {identifier for identifier, item in labels.items()
                       if getattr(item, label_name) in qualifying_labels}
            unclear = {identifier for identifier, item in labels.items()
                       if getattr(item, label_name) == "UNCLEAR"}
            if (set(criterion.qualifying_video_ids) != matches
                    or set(criterion.unclear_video_ids) != unclear):
                raise ValueError("Content counts must include exactly the saved classification labels.")

    expected_size = "NEEDS_REVIEW"
    if (
        channel.hidden_subscriber_count is not True
        and channel.subscriber_count is not None
    ):
        expected_size = (
            "PASS"
            if result.rules.min_subscribers
            <= channel.subscriber_count
            <= result.rules.max_subscribers
            else "FAIL"
        )

    if (
        result.subscribers.status != expected_size
        or result.subscribers.evaluated_at < channel.collected_at
    ):
        raise ValueError(
            "Subscriber verdict must match the saved observation and current rules."
        )

    engagement = result.engagement
    subscribers = (
        None
        if channel.hidden_subscriber_count is True
        else channel.subscriber_count
    )
    if (
        engagement.subscriber_count != subscribers
        or engagement.subscriber_observed_at != channel.collected_at
    ):
        raise ValueError(
            "Engagement subscriber inputs must match the source observation."
        )

    by_id = {video.video_id: video for video in videos}
    for counts in engagement.videos:
        video = by_id[counts.video_id]
        if (
            counts.likes,
            counts.comments,
            counts.observed_at,
        ) != (
            video.likes,
            video.comments,
            video.collected_at,
        ):
            raise ValueError(
                "Engagement video inputs must match the current saved observations."
            )

    connection.execute(
        "INSERT INTO filtering_results "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            result.filtering_id,
            result.collection_run_id,
            result.channel_id,
            result.classification_id,
            result.input_fingerprint,
            result.rules.version,
            result.status,
            result.evaluated_at.astimezone(timezone.utc).isoformat(),
            result.model_dump_json(),
        ),
    )
    return True


def list_run_filtering_results(
    connection: sqlite3.Connection, run_id: str
) -> list[FilteringStorageRecord]:
    """Return all saved evaluations for a cohort, including earlier versions."""
    rows = connection.execute(
        """SELECT record_json FROM filtering_results
        WHERE collection_run_id = ?
        ORDER BY channel_id, evaluated_at, filtering_id""",
        (run_id,),
    )
    return [
        _parse_filtering_record(row[0])
        for row in rows
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_storage() -> None:
    """Test a disposable database, never the project database."""
    now = datetime.now(timezone.utc)
    channel = ChannelRecord(
        channel_id="synthetic-channel",
        title="Synthetic storage example",
        profile_url="https://www.youtube.com/channel/synthetic-channel",
        discovery_query="synthetic storage check",
        collected_at=now,
    )
    video = VideoRecord(
        video_id="synthetic-video",
        channel_id=channel.channel_id,
        title="Synthetic storage example",
        video_url="https://www.youtube.com/watch?v=synthetic-video",
        views=0,
        collected_at=now,
    )
    result = SizeEligibilityResult(
        channel_id=channel.channel_id,
        status="NEEDS_REVIEW",
        reasons=["Synthetic example: subscriber count is unknown."],
        evaluated_at=now,
    )

    with TemporaryDirectory(prefix="outreach-storage-check-") as directory:
        test_path = Path(directory) / "test.db"
        with open_database(test_path) as connection:
            save_channel(connection, channel)
            save_video(connection, video)
            save_size_result(connection, result)

        with open_database(test_path) as connection:
            _require(
                get_channel(connection, channel.channel_id) == channel,
                "Channel round trip failed.",
            )
            _require(
                get_video(connection, video.video_id) == video,
                "Video round trip failed.",
            )
            _require(
                get_size_result(connection, channel.channel_id) == result,
                "Size result lost.",
            )
            save_channel(connection, channel)
            save_video(connection, video)
            _require(
                record_counts(connection) == {"channels": 1, "videos": 1},
                "Duplicate records created.",
            )
            _require(
                get_size_result(connection, channel.channel_id) == result,
                "Identical rerun cleared decision.",
            )

        try:
            with open_database(test_path) as connection:
                changed = channel.model_dump()
                changed["title"] = "Synthetic rolled-back update"
                save_channel(
                    connection, ChannelRecord.model_validate(changed)
                )
                orphan = video.model_dump()
                orphan.update(
                    video_id="synthetic-orphan",
                    channel_id="missing-channel",
                )
                save_video(
                    connection, VideoRecord.model_validate(orphan)
                )
        except sqlite3.IntegrityError:
            pass
        else:
            raise RuntimeError("Foreign key validation failed.")

        with open_database(test_path) as connection:
            _require(
                get_channel(connection, channel.channel_id) == channel,
                "Transaction rollback failed.",
            )
            _require(
                get_size_result(connection, channel.channel_id) == result,
                "Rolled-back decision lost.",
            )
            _require(
                record_counts(connection) == {"channels": 1, "videos": 1},
                "Rollback left extra rows.",
            )

    print(
        "Storage checks OK: persistence, duplicate IDs, missing/zero counts, "
        "foreign keys, rollback."
    )


def verify_collection_storage() -> None:
    now = datetime.now(timezone.utc)
    channel = ChannelRecord(
        channel_id="synthetic-channel",
        title="Synthetic collection storage example",
        profile_url="https://www.youtube.com/channel/synthetic-channel",
        discovery_query="synthetic check",
        collected_at=now,
    )
    video = VideoRecord(
        video_id="synthetic-video",
        channel_id=channel.channel_id,
        video_url="https://www.youtube.com/watch?v=synthetic-video",
        title="Synthetic example",
        published_at=now,
        collected_at=now,
        privacy_status="public",
        live_broadcast_content="none",
        views=0,
    )
    result = CollectionResult(
        run_id="synthetic-run",
        channel_id=channel.channel_id,
        requested_video_count=1,
        selected_video_ids=[video.video_id],
        status="COMPLETE",
        started_at=now,
        finished_at=now,
    )

    with TemporaryDirectory(
        prefix="outreach-collection-storage-check-"
    ) as directory:
        path = Path(directory) / "test.db"
        with open_database(path) as connection:
            save_channel(connection, channel)
            save_video(connection, video)
            _require(
                save_collection_result(connection, result),
                "Collection result was not saved.",
            )
            _require(
                not save_collection_result(connection, result),
                "Identical result was duplicated.",
            )

        with open_database(path) as connection:
            _require(
                get_collection_result(
                    connection, result.run_id, channel.channel_id
                ) == result,
                "Collection round trip failed.",
            )
            refreshed = video.model_dump()
            refreshed["views"] = 100
            save_video(
                connection, VideoRecord.model_validate(refreshed)
            )
            _require(
                get_collection_videos(
                    connection, result.run_id, channel.channel_id
                ) == [VideoRecord.model_validate(refreshed)],
                "Sample links did not return the current video metadata.",
            )
            _require(
                get_collection_result(
                    connection, result.run_id, channel.channel_id
                ) == result,
                "Refreshing metadata changed the sample membership.",
            )
            _require(
                len(
                    list_collection_results(
                        connection, channel.channel_id
                    )
                ) == 1,
                "Collection history duplicated.",
            )

        try:
            with open_database(path) as connection:
                uncommitted = video.model_dump()
                uncommitted["video_id"] = "synthetic-uncommitted"
                save_video(
                    connection, VideoRecord.model_validate(uncommitted)
                )
                bad = result.model_dump()
                bad.update(
                    run_id="synthetic-failed-run",
                    selected_video_ids=["missing-video"],
                )
                save_collection_result(
                    connection, CollectionResult.model_validate(bad)
                )
        except ValueError:
            pass
        else:
            raise RuntimeError("Missing sample video was accepted.")

        with open_database(path) as connection:
            _require(
                get_video(connection, "synthetic-uncommitted") is None,
                "Collection transaction did not roll back.",
            )

    print(
        "Collection storage checks OK: sample IDs, current video metadata, "
        "duplicate prevention, and rollback."
    )


def verify_filtering_storage() -> None:
    """Synthetic checks only: migration, reuse, source links, and rollback."""
    from schemas import (
        ClassificationResponse,
        VideoClassification,
        TextEvidence,
        CriterionResult,
        EngagementResult,
        EngagementVideoCounts,
    )

    observed = datetime.now(timezone.utc)
    channel = ChannelRecord(
        channel_id="synthetic-filter-channel",
        title="Synthetic channel",
        profile_url=(
            "https://www.youtube.com/channel/synthetic-filter-channel"
        ),
        subscriber_count=10000,
        hidden_subscriber_count=False,
        discovery_query="synthetic",
        collected_at=observed,
    )
    videos = [
        VideoRecord(
            video_id=f"synthetic-filter-{number}",
            channel_id=channel.channel_id,
            video_url=(
                f"https://www.youtube.com/watch?v=synthetic-filter-{number}"
            ),
            title="Python project example",
            published_at=observed,
            collected_at=observed,
            privacy_status="public",
            live_broadcast_content="none",
            likes=150,
            comments=20,
        )
        for number in range(10)
    ]
    identifiers = [video.video_id for video in videos]
    collection = CollectionResult(
        run_id="synthetic-filter-run",
        channel_id=channel.channel_id,
        selected_video_ids=identifiers,
        status="COMPLETE",
        started_at=observed,
        finished_at=observed,
    )
    evidence = [
        TextEvidence(source="title", quote="Python project")
    ]
    classification = ClassificationRecord(
        classification_id="synthetic-classification",
        collection_run_id=collection.run_id,
        channel_id=channel.channel_id,
        sample_video_ids=identifiers,
        input_fingerprint="a" * 64,
        model="synthetic-model",
        prompt_version="v1",
        criteria_version="python_course_v1",
        classified_at=observed,
        response=ClassificationResponse(
            channel_id=channel.channel_id,
            channel_niche_reason="Synthetic Python examples.",
            channel_description_evidence=[],
            videos=[
                VideoClassification(
                    video_id=identifier,
                    technology_label="TECHNOLOGY",
                    technology_reason="Synthetic programming evidence.",
                    technology_evidence=evidence,
                    relevance_label="RELATED",
                    relevance_reason="Synthetic Python relevance.",
                    relevance_evidence=evidence,
                )
                for identifier in identifiers
            ],
        ),
    )
    criterion = CriterionResult(
        status="PASS",
        reasons=["Synthetic ten matches."],
        qualifying_video_ids=identifiers,
    )
    filtering = ChannelFilteringResult(
        filtering_id="synthetic-filtering",
        collection_run_id=collection.run_id,
        channel_id=channel.channel_id,
        sample_video_ids=identifiers,
        classification_id=classification.classification_id,
        input_fingerprint="b" * 64,
        technology=criterion,
        content_relevance=criterion,
        subscribers=SizeEligibilityResult(
            channel_id=channel.channel_id,
            status="PASS",
            reasons=["Synthetic size."],
            evaluated_at=observed,
        ),
        engagement=EngagementResult(
            subscriber_count=10000,
            subscriber_observed_at=observed,
            videos=[
                EngagementVideoCounts(
                    video_id=v.video_id,
                    likes=v.likes,
                    comments=v.comments,
                    observed_at=observed,
                )
                for v in videos
            ],
            rate_percent="1.7",
            status="PASS",
            reasons=["Synthetic arithmetic only."],
        ),
        status="PASS",
        reasons=["Synthetic complete checks."],
        evaluated_at=observed,
    )

    with TemporaryDirectory(
        prefix="outreach-filtering-storage-check-"
    ) as directory:
        path = Path(directory) / "test.db"

        # Build version 2, then exercise migration and backup.
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            _create_initial_tables(connection)
            _create_collection_tables(connection)
            connection.execute("PRAGMA user_version = 2")
            save_channel(connection, channel)
            for video in videos:
                save_video(connection, video)
            save_collection_result(connection, collection)

        with open_database(path) as connection:
            _require(
                record_counts(connection) == {"channels": 1, "videos": 10},
                "Migration lost records.",
            )
            _require(
                list_run_collection_results(
                    connection, collection.run_id
                ) == [collection],
                "Cohort lost.",
            )
            _require(
                save_classification(connection, classification),
                "Classification not saved.",
            )
            _require(
                not save_classification(connection, classification),
                "Classification duplicated.",
            )
            _require(
                save_filtering_result(connection, filtering),
                "Filtering not saved.",
            )
            _require(
                not save_filtering_result(connection, filtering),
                "Filtering duplicated.",
            )

        backups = list(path.parent.glob("test.v2-backup-*.db"))
        _require(
            len(backups) == 1,
            "Migration backup missing or repeated.",
        )
        with closing(sqlite3.connect(backups[0])) as backup:
            _require(
                backup.execute("PRAGMA user_version").fetchone()[0] == 2,
                "Wrong backup version.",
            )

        lookup = dict(
            channel_id=channel.channel_id,
            input_fingerprint="a" * 64,
            model="synthetic-model",
            prompt_version="v1",
            criteria_version="python_course_v1",
            sample_video_ids=identifiers,
        )
        with open_database(path) as connection:
            _require(
                find_reusable_classification(
                    connection, **lookup
                ) == classification,
                "Reuse lookup failed.",
            )
            for name, value in (
                ("input_fingerprint", "c" * 64),
                ("model", "other-model"),
                ("prompt_version", "v2"),
                ("criteria_version", "v2"),
                ("sample_video_ids", identifiers[::-1]),
            ):
                _require(
                    find_reusable_classification(
                        connection, **{**lookup, name: value}
                    ) is None,
                    "Changed input or version reused a classification.",
                )

            _require(
                get_filtering_result(
                    connection, filtering.filtering_id
                ) == filtering,
                "Filtering round trip failed.",
            )
            _require(
                list_run_filtering_results(
                    connection, collection.run_id
                ) == [filtering],
                "Cohort filtering failed.",
            )

            # A new cohort reuses the judgment without another LLM record.
            another = CollectionResult.model_validate(
                {
                    **collection.model_dump(),
                    "run_id": "synthetic-rerun",
                }
            )
            save_collection_result(connection, another)
            repeated = ChannelFilteringResult.model_validate(
                {
                    **filtering.model_dump(),
                    "filtering_id": "synthetic-reused-filter",
                    "collection_run_id": another.run_id,
                }
            )
            save_filtering_result(connection, repeated)
            _require(
                connection.execute(
                    "SELECT COUNT(*) FROM classifications"
                ).fetchone()[0] == 1,
                "Cross-run reuse duplicated classification.",
            )

        bad_values = classification.model_dump()
        bad_values.update(
            classification_id="invented-evidence",
            input_fingerprint="d" * 64,
        )
        bad_values["response"]["videos"][0]["technology_evidence"][0][
            "quote"
        ] = "invented quote"

        try:
            with open_database(path) as connection:
                transient = VideoRecord.model_validate(
                    {**videos[0].model_dump(), "views": 999}
                )
                save_video(connection, transient)
                save_classification(
                    connection,
                    ClassificationRecord.model_validate(bad_values),
                )
        except ValueError:
            pass
        else:
            raise RuntimeError("Invented evidence was accepted.")

        with open_database(path) as connection:
            _require(
                get_video(connection, videos[0].video_id).views is None,
                "Failed evaluation did not roll back.",
            )
            for changed in (
                {**classification.model_dump(), "model": "changed"},
                {
                    **classification.model_dump(),
                    "classification_id": "duplicate-cache-key",
                },
            ):
                try:
                    save_classification(
                        connection,
                        ClassificationRecord.model_validate(changed),
                    )
                except (ValueError, sqlite3.IntegrityError):
                    pass
                else:
                    raise RuntimeError(
                        "An immutable record or duplicate cache key was accepted."
                    )

            _require(
                connection.execute(
                    "PRAGMA integrity_check"
                ).fetchone()[0] == "ok",
                "Integrity check failed.",
            )
            _require(
                not connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall(),
                "Foreign key check failed.",
            )

    print(
        "Filtering storage checks OK: migration backup, evidence, cohort links, "
        "reuse, duplicate prevention, and rollback."
    )


def verify_title_storage() -> None:
    """Synthetic persistence checks for the description/title policy."""
    from schemas import (
        ChannelNicheClassification, TitleRelevanceClassification,
        TitleClassificationResponse, ChannelNicheResult, CriterionResult,
        EngagementResult, EngagementVideoCounts,
        ClassificationResponse, VideoClassification, TextEvidence,
    )

    observed = datetime.now(timezone.utc)
    channel = ChannelRecord(
        channel_id="synthetic-title-channel", title="Synthetic programming channel",
        profile_url="https://www.youtube.com/channel/synthetic-title-channel",
        description="Software development and Python tutorials.",
        subscriber_count=10000, hidden_subscriber_count=False,
        discovery_query="synthetic", collected_at=observed,
    )
    videos = [VideoRecord(
        video_id=f"synthetic-title-{number}", channel_id=channel.channel_id,
        video_url=f"https://www.youtube.com/watch?v=synthetic-title-{number}",
        title="Python project" if number < 5 else "Baking bread",
        description=None, published_at=observed, collected_at=observed,
        privacy_status="public", live_broadcast_content="none", views=0, likes=150, comments=20,
    ) for number in range(10)]
    identifiers = [video.video_id for video in videos]
    collection = CollectionResult(
        run_id="synthetic-title-run", channel_id=channel.channel_id,
        selected_video_ids=identifiers, status="COMPLETE", started_at=observed, finished_at=observed,
    )
    response = TitleClassificationResponse(
        channel_id=channel.channel_id,
        niche=ChannelNicheClassification(label="TECHNOLOGY",
            reason="The supplied description concerns software development.", evidence_source="channel_description"),
        videos=[TitleRelevanceClassification(video_id=video.video_id,
            relevance_label="RELATED" if number < 5 else "NO_MATCH",
            relevance_reason="Synthetic title-only judgment.", evidence_source="title")
            for number, video in enumerate(videos)],
    )
    classification = TitleClassificationRecord(
        classification_id="synthetic-title-classification", collection_run_id=collection.run_id,
        channel_id=channel.channel_id, sample_video_ids=identifiers,
        input_fingerprint="a" * 64, model="synthetic-model", prompt_version="synthetic-prompt",
        response=response, classified_at=observed,
    )
    legacy = ClassificationRecord(
        classification_id="synthetic-legacy-classification", collection_run_id=collection.run_id,
        channel_id=channel.channel_id, sample_video_ids=identifiers,
        input_fingerprint=classification.input_fingerprint, model=classification.model,
        prompt_version=classification.prompt_version, criteria_version="python_course_v1",
        classified_at=observed,
        response=ClassificationResponse(channel_id=channel.channel_id,
            channel_niche_reason="Synthetic historical judgment.", channel_description_evidence=[],
            videos=[VideoClassification(video_id=video.video_id,
                technology_label="TECHNOLOGY" if number < 5 else "OTHER",
                technology_reason="Synthetic historical title judgment.",
                technology_evidence=[TextEvidence(source="title", quote=video.title)],
                relevance_label="RELATED" if number < 5 else "NO_MATCH",
                relevance_reason="Synthetic historical relevance judgment.",
                relevance_evidence=[TextEvidence(source="title", quote=video.title)])
                for number, video in enumerate(videos)]),
    )
    relevance = CriterionResult(status="PASS", reasons=["Five related titles."], qualifying_video_ids=identifiers[:5])
    size = SizeEligibilityResult(channel_id=channel.channel_id, status="PASS",
        reasons=["Synthetic size."], evaluated_at=observed)
    engagement = EngagementResult(subscriber_count=10000, subscriber_observed_at=observed,
        videos=[EngagementVideoCounts(video_id=video.video_id, likes=video.likes,
            comments=video.comments, observed_at=observed) for video in videos],
        rate_percent="1.7", status="PASS", reasons=["Synthetic arithmetic only."])
    filtering = TitleChannelFilteringResult(
        filtering_id="synthetic-title-filtering", collection_run_id=collection.run_id,
        channel_id=channel.channel_id, sample_video_ids=identifiers,
        classification_id=classification.classification_id, input_fingerprint="b" * 64,
        technology=ChannelNicheResult(label="TECHNOLOGY", description_available=True,
            evidence_source="channel_description", status="PASS", reasons=["Description-based niche."]),
        content_relevance=relevance, subscribers=size, engagement=engagement,
        status="PASS", reasons=["Synthetic combined verdict."], evaluated_at=observed,
    )
    historical_filter = ChannelFilteringResult(
        filtering_id="synthetic-legacy-filtering", collection_run_id=collection.run_id,
        channel_id=channel.channel_id, sample_video_ids=identifiers,
        classification_id=legacy.classification_id, input_fingerprint="b" * 64,
        technology=relevance, content_relevance=relevance, subscribers=size, engagement=engagement,
        status="PASS", reasons=["Synthetic historical verdict."], evaluated_at=observed,
    )
    lookup = dict(channel_id=channel.channel_id, input_fingerprint=classification.input_fingerprint,
        model=classification.model, prompt_version=classification.prompt_version,
        criteria_version="python_course_v2", sample_video_ids=identifiers)
    with TemporaryDirectory(prefix="outreach-title-storage-check-") as directory:
        path = Path(directory) / "test.db"
        with open_database(path) as connection:
            save_channel(connection, channel)
            for video in videos:
                save_video(connection, video)
            save_collection_result(connection, collection)
            save_classification(connection, legacy)
            _require(find_reusable_classification(connection, **lookup) is None,
                     "Old classification reused under new criteria.")
            _require(save_classification(connection, classification), "New classification not saved.")
            _require(not save_classification(connection, classification), "New classification duplicated.")
            _require(get_classification(connection, legacy.classification_id) == legacy, "Historical record changed.")
            _require(get_classification(connection, classification.classification_id) == classification,
                     "New classification round trip failed.")
            _require(find_reusable_classification(connection, **lookup) == classification, "New reuse failed.")
            _require(find_reusable_classification(connection, **{**lookup, "sample_video_ids": identifiers[::-1]}) is None,
                     "Changed sample order reused classification.")
            save_filtering_result(connection, historical_filter)
            _require(save_filtering_result(connection, filtering), "New filtering not saved.")
            _require(not save_filtering_result(connection, filtering), "New filtering duplicated.")
            records = list_run_filtering_results(connection, collection.run_id)
            _require(len(records) == 2 and historical_filter in records and filtering in records,
                     "Mixed-format filtering records not readable.")
            _require(get_channel(connection, channel.channel_id).description == channel.description,
                     "Channel description was not preserved.")
            another = CollectionResult.model_validate({**collection.model_dump(), "run_id": "synthetic-title-rerun"})
            save_collection_result(connection, another)
            repeated = TitleChannelFilteringResult.model_validate({**filtering.model_dump(),
                "filtering_id": "synthetic-title-reused-filter", "collection_run_id": another.run_id})
            save_filtering_result(connection, repeated)
            _require(connection.execute("SELECT COUNT(*) FROM classifications").fetchone()[0] == 2,
                     "Cross-run filtering duplicated a classification.")

        def reject_filter(data):
            try:
                with open_database(path) as connection:
                    save_filtering_result(connection, TitleChannelFilteringResult.model_validate(data))
            except ValueError:
                return
            raise RuntimeError("A mismatched title-only filtering record was accepted.")

        bad = filtering.model_dump()
        bad.update(filtering_id="bad-label-counts", input_fingerprint="c" * 64)
        bad["content_relevance"]["qualifying_video_ids"] = identifiers[1:6]
        reject_filter(bad)
        bad = filtering.model_dump()
        bad.update(filtering_id="bad-niche", input_fingerprint="c" * 64, status="FAIL")
        bad["technology"].update(label="OTHER", status="FAIL")
        reject_filter(bad)
        bad = filtering.model_dump()
        bad.update(filtering_id="old-classification-new-policy", input_fingerprint="c" * 64,
                   classification_id=legacy.classification_id)
        reject_filter(bad)
        bad = filtering.model_dump()
        bad.update(filtering_id="bad-observation", input_fingerprint="c" * 64)
        bad["engagement"]["videos"][0]["observed_at"] = "2000-01-01T00:00:00Z"
        reject_filter(bad)

        with open_database(path) as connection:
            save_channel(connection, ChannelRecord.model_validate({**channel.model_dump(), "description": None}))
        bad_class = TitleClassificationRecord.model_validate({**classification.model_dump(),
            "classification_id": "missing-description-definite", "input_fingerprint": "d" * 64})
        try:
            with open_database(path) as connection:
                save_video(connection, VideoRecord.model_validate({**videos[0].model_dump(), "views": 999}))
                save_classification(connection, bad_class)
        except ValueError:
            pass
        else:
            raise RuntimeError("Missing description accepted as definite niche.")
        review_data = classification.model_dump()
        review_data.update(classification_id="synthetic-missing-description", input_fingerprint="d" * 64)
        review_data["response"]["niche"].update(label="UNCLEAR", evidence_source="none", reason="No description available.")
        review_record = TitleClassificationRecord.model_validate(review_data)
        review_filter_data = filtering.model_dump()
        review_filter_data.update(filtering_id="synthetic-missing-description-filter", input_fingerprint="e" * 64,
            classification_id=review_record.classification_id, status="NEEDS_REVIEW")
        review_filter_data["technology"].update(label="UNCLEAR", description_available=False,
            evidence_source="none", status="NEEDS_REVIEW", reasons=["Description missing."])
        review_filter = TitleChannelFilteringResult.model_validate(review_filter_data)
        with open_database(path) as connection:
            _require(get_video(connection, identifiers[0]).views == 0, "Invalid classification did not roll back video write.")
            save_classification(connection, review_record)
            save_filtering_result(connection, review_filter)
            _require(get_filtering_result(connection, review_filter.filtering_id) == review_filter,
                     "Missing-description review round trip failed.")
            try:
                _parse_classification_record({**legacy.model_dump(), "criteria_version": "python_course_v2"})
            except ValueError:
                pass
            else:
                raise RuntimeError("Historical shape was relabelled as new policy.")
            _require(connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION,
                     "Unnecessary SQL migration occurred.")
            _require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "Storage integrity failed.")
            _require(not connection.execute("PRAGMA foreign_key_check").fetchall(), "Foreign keys failed.")
    print("Title-only storage checks OK: new and legacy formats, channel descriptions, niche decisions, title IDs, version-separated reuse, duplicate prevention, missing-description review, and rollback.")


def verify_enrichment_evidence_storage() -> None:
    stamp = datetime.now(timezone.utc)
    channel = ChannelRecord(channel_id="evidence-check", title="Synthetic",
        profile_url="https://www.youtube.com/channel/evidence-check",
        discovery_query="synthetic", collected_at=stamp)
    source = {"channel_id": channel.channel_id, "source_id": "channel", "kind": "CHANNEL_DESCRIPTION",
        "url": str(channel.profile_url), "text": "Business: creator@example.com",
        "observed_at": stamp.isoformat(), "status": "AVAILABLE", "reason": None}
    with TemporaryDirectory(prefix="outreach-evidence-check-") as directory:
        path = Path(directory) / "test.db"
        with open_database(path) as connection:
            save_channel(connection, channel)
            # Simulate the prior version plus existing enrichment history.
            connection.execute("CREATE TABLE profile_enrichments (record_json TEXT)")
            connection.execute("INSERT INTO profile_enrichments VALUES ('historical-profile')")
            connection.execute("DROP TABLE enrichment_batch_sources")
            connection.execute("DROP TABLE enrichment_batches")
            connection.execute("DROP TABLE enrichment_sources")
            connection.execute("PRAGMA user_version = 3")
        with open_database(path) as connection:
            _require(connection.execute("PRAGMA user_version").fetchone()[0] == 4, "Evidence migration failed.")
            _require(get_channel(connection, channel.channel_id) == channel, "Migration changed creator records.")
            _require(connection.execute("SELECT record_json FROM profile_enrichments").fetchone()[0]
                     == "historical-profile", "Migration changed enrichment history.")
            identifier = save_enrichment_source(connection, source)
            _require(save_enrichment_source(connection, source) == identifier, "Evidence reuse failed.")
            _require(get_enrichment_source(connection, identifier) == source, "Evidence round trip failed.")
            candidate = {"kind": "EMAIL", "value": "creator@example.com", "evidence_ids": [identifier]}
            context = {"pipeline_version": "synthetic-v1", "stage": "descriptions"}
            batch_id = save_enrichment_batch(connection, channel.channel_id, [identifier], [candidate], context)
            _require(save_enrichment_batch(connection, channel.channel_id, [identifier], [candidate], context)
                     == batch_id, "Batch reuse failed.")
            _require(get_enrichment_batch(connection, batch_id)["candidates"] == [candidate], "Batch lost evidence links.")
            for candidates in ([{**candidate, "value": "invented@example.com"}], [candidate, candidate],
                               [{**candidate, "evidence_ids": ["missing"]}]):
                try:
                    save_enrichment_batch(connection, channel.channel_id, [identifier], candidates, context)
                except ValueError:
                    pass
                else:
                    raise RuntimeError("Invalid candidate evidence accepted.")
            try:
                save_enrichment_source(connection, {**source, "observed_at": "2026-01-01T00:00:00"})
            except ValueError:
                pass
            else:
                raise RuntimeError("Naive evidence timestamp accepted.")
            second = {**source, "source_id": "video:example", "kind": "VIDEO_DESCRIPTION"}
            second_id = save_enrichment_source(connection, second)
            merged = {**candidate, "evidence_ids": [identifier, second_id]}
            save_enrichment_batch(connection, channel.channel_id, [identifier, second_id], [merged], context)
            _require(connection.execute("SELECT COUNT(*) FROM enrichment_sources").fetchone()[0] == 2,
                     "Repeated sources created duplicate rows.")
            before = connection.execute("SELECT COUNT(*) FROM enrichment_batches").fetchone()[0]
        try:
            with open_database(path) as connection:
                save_enrichment_batch(connection, channel.channel_id, [identifier], [], {"pipeline_version": "rollback"})
                raise ValueError("Synthetic rollback")
        except ValueError:
            pass
        with open_database(path) as connection:
            _require(connection.execute("SELECT COUNT(*) FROM enrichment_batches").fetchone()[0] == before,
                     "Batch rollback failed.")
            _require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "Evidence integrity failed.")
            _require(not connection.execute("PRAGMA foreign_key_check").fetchall(), "Evidence foreign keys failed.")
        backups = list(path.parent.glob("test.v3-backup-*.db"))
        _require(len(backups) == 1, "Version 3 backup missing or repeated.")
        with closing(sqlite3.connect(backups[0])) as backup:
            _require(backup.execute("PRAGMA user_version").fetchone()[0] == 3, "Backup version changed.")
    print("Enrichment evidence storage checks OK: migration backup, existing history, sources, candidate links, reuse, validation and rollback.")


def main() -> None:
    load_environment()
    path = database_path()
    verify_storage()
    verify_collection_storage()
    verify_filtering_storage()
    verify_title_storage()
    verify_enrichment_evidence_storage()

    with open_database(path) as connection:
        counts = record_counts(connection)
        collection_count = connection.execute(
            "SELECT COUNT(*) FROM collection_results"
        ).fetchone()[0]
        schema_version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
        classification_count = connection.execute(
            "SELECT COUNT(*) FROM classifications"
        ).fetchone()[0]
        filtering_count = connection.execute(
            "SELECT COUNT(*) FROM filtering_results"
        ).fetchone()[0]

    print(f"Database ready: {path}")
    print(
        f"Stored channels: {counts['channels']}; "
        f"stored videos: {counts['videos']}"
    )
    print(
        f"Database schema version: {schema_version}; "
        f"collection results: {collection_count}"
    )
    print(
        f"Stored classifications: {classification_count}; "
        f"filtering results: {filtering_count}"
    )
    print(
        "Synthetic examples used a temporary database; "
        "no API calls were made."
    )


if __name__ == "__main__":
    main()

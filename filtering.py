"""Filter a named collection cohort using saved metadata and classifications.

No network calls. --preview calculates without writes. --self-check uses only
synthetic inputs and temporary SQLite storage. Existing evaluations are reused
when their complete input fingerprint matches; changed inputs create history.
"""

from config import database_path, load_environment
import argparse
from collections import Counter
from contextlib import closing
from decimal import Decimal, localcontext
import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from uuid import uuid4

from classification import (
    DEFAULT_MODEL, ClassificationError, cached_record, canonical, now,
    prepare_input, read_cohort,
)
from database import (
    get_channel, get_collection_videos, list_run_filtering_results,
    open_database, save_filtering_result,
)
from schemas import (
    ChannelNicheResult, CriterionResult, EngagementResult,
    EngagementVideoCounts, SizeEligibilityResult, TitleChannelFilteringResult,
    TitleClassificationRecord, TitleFilteringRules,
)


RULES = TitleFilteringRules()


def evaluate(run_id, channel, videos, classification, rules=RULES):
    """Calculate with Decimal; preserve missing values and exact sample IDs."""
    observed = now()
    ids = [video.video_id for video in videos]

    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("Filtering requires ten unique sample videos.")

    if any(video.channel_id != channel.channel_id for video in videos):
        raise ValueError("Video ownership does not match the channel.")

    if max([channel.collected_at] + [v.collected_at for v in videos]) > observed:
        raise ValueError("Source observations cannot be in the future.")

    available = bool((channel.description or "").strip())
    matches, unclear = [], []

    if classification is None:
        niche = ChannelNicheResult(
            description_available=available,
            evidence_source="none",
            status="NEEDS_REVIEW",
            reasons=[
                "No matching current classification; run classification.py first."
            ],
        )
        relevance_status = "NEEDS_REVIEW"
        relevance_reason = (
            "No matching current title classification is available."
        )
    else:
        if (
            not isinstance(classification, TitleClassificationRecord)
            or classification.criteria_version != rules.version
            or classification.sample_video_ids != ids
            or classification.classified_at > observed
        ):
            raise ValueError(
                "Classification format, sample, version or timestamp is invalid."
            )

        classification.response.validate_sources(channel, videos)
        judgment = classification.response.niche

        niche = ChannelNicheResult(
            label=judgment.label,
            description_available=available,
            evidence_source=judgment.evidence_source,
            status={
                "TECHNOLOGY": "PASS",
                "OTHER": "FAIL",
            }.get(judgment.label, "NEEDS_REVIEW"),
            reasons=[judgment.reason],
        )

        labels = {
            item.video_id: item.relevance_label
            for item in classification.response.videos
        }

        matches = [
            identifier
            for identifier in ids
            if labels[identifier] in ("MATCH", "RELATED")
        ]
        unclear = [
            identifier
            for identifier in ids
            if labels[identifier] == "UNCLEAR"
        ]

        relevance_status = (
            "PASS"
            if len(matches) >= rules.relevance_min_videos
            else "NEEDS_REVIEW"
            if len(matches) + len(unclear) >= rules.relevance_min_videos
            else "FAIL"
        )

        relevance_reason = (
            f"{len(matches)}/10 MATCH or RELATED titles; "
            f"{len(unclear)} unclear; "
            f"at least {rules.relevance_min_videos} qualifying titles required."
        )

    relevance = CriterionResult(
        status=relevance_status,
        reasons=[relevance_reason],
        qualifying_video_ids=matches,
        unclear_video_ids=unclear,
    )

    subscribers = (
        None
        if channel.hidden_subscriber_count is True
        else channel.subscriber_count
    )

    size_status = (
        "NEEDS_REVIEW"
        if subscribers is None
        else "PASS"
        if rules.min_subscribers <= subscribers <= rules.max_subscribers
        else "FAIL"
    )

    size = SizeEligibilityResult(
        channel_id=channel.channel_id,
        status=size_status,
        reasons=[
            "Subscriber count is missing or hidden."
            if subscribers is None
            else (
                f"Saved subscribers: {subscribers}; inclusive range "
                f"{rules.min_subscribers}-{rules.max_subscribers}."
            )
        ],
        evaluated_at=observed,
    )

    counts = [
        EngagementVideoCounts(
            video_id=v.video_id,
            likes=v.likes,
            comments=v.comments,
            observed_at=v.collected_at,
        )
        for v in videos
    ]

    missing = [
        v.video_id
        for v in videos
        if v.likes is None or v.comments is None
    ]

    rate, engagement_status = None, "NEEDS_REVIEW"

    if subscribers is not None and subscribers > 0 and not missing:
        interactions = sum(v.likes + v.comments for v in videos)
        numerator = Decimal(100 * interactions)
        denominator = Decimal(10 * subscribers)

        with localcontext() as context:
            context.prec = 28
            rate = numerator / denominator

        engagement_status = (
            "PASS"
            if numerator > rules.engagement_threshold_percent * denominator
            else "FAIL"
        )

        engagement_reason = (
            f"{interactions} interactions across ten videos; rate {rate}%; "
            f"must be strictly greater than "
            f"{rules.engagement_threshold_percent}%."
        )
    else:
        engagement_reason = (
            "Cannot calculate: subscriber count is missing, hidden or zero. "
            if subscribers is None or subscribers == 0
            else ""
        )

        if missing:
            engagement_reason += (
                "Missing likes/comments for video IDs: "
                + ", ".join(missing)
                + "."
            )

    engagement = EngagementResult(
        subscriber_count=subscribers,
        subscriber_observed_at=channel.collected_at,
        videos=counts,
        rate_percent=rate,
        threshold_percent=rules.engagement_threshold_percent,
        status=engagement_status,
        reasons=[engagement_reason],
    )

    statuses = {
        "niche": niche.status,
        "relevance": relevance.status,
        "subscribers": size.status,
        "engagement": engagement.status,
    }

    verdict = (
        "FAIL"
        if "FAIL" in statuses.values()
        else "NEEDS_REVIEW"
        if "NEEDS_REVIEW" in statuses.values()
        else "PASS"
    )

    identity = {
        "engine": "filtering_v1",
        "run_id": run_id,
        "rules": rules.model_dump(mode="json"),
        "classification_id": (
            classification.classification_id if classification else None
        ),
        "classification_fingerprint": (
            classification.input_fingerprint if classification else None
        ),
        "channel_description": channel.description,
        "hidden_subscribers": channel.hidden_subscriber_count,
        "subscribers": subscribers,
        "subscriber_observed_at": channel.collected_at.isoformat(),
        "videos": [
            {
                "id": v.video_id,
                "title": v.title,
                "likes": v.likes,
                "comments": v.comments,
                "observed_at": v.collected_at.isoformat(),
            }
            for v in videos
        ],
    }

    fingerprint = hashlib.sha256(
        canonical(identity).encode("utf-8")
    ).hexdigest()

    return TitleChannelFilteringResult(
        filtering_id=uuid4().hex,
        collection_run_id=run_id,
        channel_id=channel.channel_id,
        sample_video_ids=ids,
        classification_id=(
            classification.classification_id if classification else None
        ),
        input_fingerprint=fingerprint,
        rules=rules,
        technology=niche,
        content_relevance=relevance,
        subscribers=size,
        engagement=engagement,
        status=verdict,
        reasons=[
            "; ".join(
                f"{name}: {status}"
                for name, status in statuses.items()
            )
        ],
        evaluated_at=observed,
    )


def filter_cohort(
    path, run_id, limit, model=DEFAULT_MODEL, preview=False
):
    total, items = read_cohort(path, run_id, limit)

    print(
        f"Collection run: {run_id}; cohort channels: {total}; "
        f"selected: {len(items)}/{limit}."
    )
    print(
        "Saved channel descriptions, titles and statistics only; "
        "no API or LLM calls."
    )
    print(
        "Observation ages are reported; no automatic freshness cutoff."
    )

    with closing(
        sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro",
            uri=True,
        )
    ) as connection:
        connection.row_factory = sqlite3.Row
        previous = list_run_filtering_results(connection, run_id)

    outcomes, skipped, reused = [], 0, 0

    for collection, channel, videos, problem in items:
        if problem:
            print(f"SKIPPED {collection.channel_id}: {problem}")
            skipped += 1
            continue

        fingerprint = prepare_input(channel, videos, model)[1]

        classification = cached_record(
            path, channel, videos, fingerprint, model
        )

        decision = evaluate(
            run_id, channel, videos, classification
        )

        existing = next(
            (
                r
                for r in previous
                if isinstance(r, TitleChannelFilteringResult)
                and r.channel_id == channel.channel_id
                and r.input_fingerprint == decision.input_fingerprint
            ),
            None,
        )

        if existing:
            decision = existing
            reused += 1
        elif not preview:
            with open_database(path) as connection:
                fresh_channel = get_channel(
                    connection, channel.channel_id
                )
                fresh_videos = get_collection_videos(
                    connection, run_id, channel.channel_id
                )
                fresh_fingerprint = prepare_input(
                    fresh_channel, fresh_videos, model
                )[1]

                if (
                    classification is not None
                    and fresh_fingerprint
                    != classification.input_fingerprint
                ):
                    raise ValueError(
                        "Classification source changed; "
                        "rerun classification before filtering."
                    )

                if (
                    fresh_channel.description != channel.description
                    or fresh_channel.hidden_subscriber_count
                    != channel.hidden_subscriber_count
                ):
                    raise ValueError(
                        "Channel inputs changed during evaluation."
                    )

                save_filtering_result(connection, decision)

        outcomes.append(decision)

        oldest = min(
            [channel.collected_at]
            + [v.collected_at for v in videos]
        )
        age = (now() - oldest).total_seconds() / 86400

        action = (
            "REUSED"
            if existing
            else "PREVIEW"
            if preview
            else "SAVED"
        )

        print(
            f"\n{action} {channel.title} | {channel.channel_id}: "
            f"{decision.status}; oldest inputs {age:.1f} days"
        )

        for name, criterion in (
            ("Niche", decision.technology),
            ("Relevance", decision.content_relevance),
            ("Subscribers", decision.subscribers),
            ("Engagement", decision.engagement),
        ):
            print(
                f"  {name}: {criterion.status}. "
                f"{' '.join(criterion.reasons)}"
            )

    tally = Counter(r.status for r in outcomes)

    print(
        f"\nEvaluated: {len(outcomes)}; "
        f"PASS: {tally['PASS']}; "
        f"FAIL: {tally['FAIL']}; "
        f"NEEDS_REVIEW: {tally['NEEDS_REVIEW']}; "
        f"reused: {reused}; skipped: {skipped}."
    )

    print(
        "Derived-metric data-use permission remains "
        "NOT_ESTABLISHED; the limitation is stored."
    )
    print(
        "Preview: no writes."
        if preview
        else "Filtering results saved; earlier evaluations retained."
    )

    return 0 if len(items) == limit and not skipped else 2


def self_check():
    from schemas import (
        ChannelRecord, VideoRecord,
        TitleClassificationResponse, CollectionResult,
    )
    from database import (
        save_channel, save_video,
        save_collection_result, save_classification,
    )

    observed = now()

    channel = ChannelRecord(
        channel_id="synthetic",
        title="Synthetic",
        description="Python tutorials",
        profile_url="https://www.youtube.com/channel/synthetic",
        subscriber_count=10000,
        hidden_subscriber_count=False,
        discovery_query="synthetic",
        collected_at=observed,
    )

    videos = [
        VideoRecord(
            video_id=f"synthetic-{i}",
            channel_id=channel.channel_id,
            video_url=(
                f"https://www.youtube.com/watch?v=synthetic-{i}"
            ),
            title="Python tutorial",
            likes=150,
            comments=20,
            privacy_status="public",
            live_broadcast_content="none",
            published_at=observed,
            collected_at=observed,
        )
        for i in range(10)
    ]

    ids = [v.video_id for v in videos]

    def judgment(
        matches=5, uncertain=0, label="TECHNOLOGY"
    ):
        response = TitleClassificationResponse(
            channel_id=channel.channel_id,
            niche={
                "label": label,
                "reason": "Synthetic niche judgment",
                "evidence_source": "channel_description",
            },
            videos=[
                {
                    "video_id": identifier,
                    "relevance_label": (
                        "RELATED"
                        if i < matches
                        else "UNCLEAR"
                        if i < matches + uncertain
                        else "NO_MATCH"
                    ),
                    "relevance_reason": "Synthetic",
                    "evidence_source": "title",
                }
                for i, identifier in enumerate(ids)
            ],
        )

        return TitleClassificationRecord(
            classification_id=uuid4().hex,
            collection_run_id="check",
            channel_id=channel.channel_id,
            sample_video_ids=ids,
            input_fingerprint=prepare_input(
                channel, videos, DEFAULT_MODEL
            )[1],
            model=DEFAULT_MODEL,
            prompt_version="classification_v4",
            response=response,
            classified_at=observed,
        )

    saved = judgment()

    assert evaluate(
        "check", channel, videos, saved
    ).status == "PASS"

    for matches, unclear, expected in (
        (4, 0, "FAIL"),
        (4, 1, "NEEDS_REVIEW"),
        (3, 1, "FAIL"),
    ):
        assert evaluate(
            "check",
            channel,
            videos,
            judgment(matches, unclear),
        ).content_relevance.status == expected

    assert evaluate(
        "check", channel, videos, judgment(label="OTHER")
    ).status == "FAIL"

    boundary = [
        VideoRecord.model_validate(
            {**v.model_dump(), "likes": 120}
        )
        for v in videos
    ]

    assert evaluate(
        "check", channel, boundary, saved
    ).engagement.rate_percent == Decimal("1.4")

    assert evaluate(
        "check", channel, boundary, saved
    ).status == "FAIL"

    missing = list(videos)
    missing[0] = VideoRecord.model_validate(
        {**missing[0].model_dump(), "likes": None}
    )

    assert evaluate(
        "check", channel, missing, saved
    ).status == "NEEDS_REVIEW"

    zeros = [
        VideoRecord.model_validate(
            {**v.model_dump(), "likes": 0, "comments": 0}
        )
        for v in videos
    ]

    assert evaluate(
        "check", channel, zeros, saved
    ).engagement.rate_percent == 0

    assert evaluate(
        "check", channel, videos, None
    ).status == "NEEDS_REVIEW"

    hidden = ChannelRecord.model_validate(
        {
            **channel.model_dump(),
            "hidden_subscriber_count": True,
        }
    )

    hidden_result = evaluate(
        "check", hidden, videos, saved
    )

    assert hidden_result.subscribers.status == "NEEDS_REVIEW"
    assert hidden_result.engagement.rate_percent is None
    assert hidden_result.engagement.status == "NEEDS_REVIEW"

    empty = ChannelRecord.model_validate(
        {**channel.model_dump(), "description": None}
    )

    assert evaluate(
        "check", empty, videos, None
    ).technology.status == "NEEDS_REVIEW"

    with TemporaryDirectory(
        prefix="outreach-filter-check-"
    ) as directory:
        path = Path(directory) / "test.db"

        with open_database(path) as connection:
            save_channel(connection, channel)

            for video in videos:
                save_video(connection, video)

            save_collection_result(
                connection,
                CollectionResult(
                    run_id="check",
                    channel_id=channel.channel_id,
                    selected_video_ids=ids,
                    status="COMPLETE",
                    started_at=observed,
                    finished_at=observed,
                ),
            )
            save_classification(connection, saved)

        assert filter_cohort(
            path, "check", 1, preview=True
        ) == 0

        with open_database(path) as connection:
            assert not list_run_filtering_results(
                connection, "check"
            )

        assert filter_cohort(path, "check", 1) == 0
        assert filter_cohort(path, "check", 1) == 0

        with open_database(path) as connection:
            assert len(
                list_run_filtering_results(connection, "check")
            ) == 1

            changed = VideoRecord.model_validate(
                {**videos[0].model_dump(), "likes": 151}
            )
            save_video(connection, changed)

        assert filter_cohort(path, "check", 1) == 0

        with open_database(path) as connection:
            assert len(
                list_run_filtering_results(connection, "check")
            ) == 2

            assert connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"

        try:
            with open_database(path) as connection:
                save_video(
                    connection,
                    VideoRecord.model_validate(
                        {
                            **videos[0].model_dump(),
                            "likes": 999,
                        }
                    ),
                )
                raise ValueError("Synthetic rollback check")
        except ValueError:
            pass

        with open_database(path) as connection:
            assert get_collection_videos(
                connection, "check", channel.channel_id
            )[0].likes == 151

    print(
        "Filtering checks OK: thresholds, strict 1.4 boundary, "
        "missing/zero inputs, verdicts, preview, persistence, "
        "reuse and changed-count history."
    )
    print(
        "Synthetic data and temporary storage only; "
        "no network requests or project database changes."
    )


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=(
            DEFAULT_MODEL,
            "openai/gpt-oss-20b",
        ),
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 5000:
        parser.error("--limit must be between 1 and 5000.")

    if args.self_check:
        self_check()
        return 0

    if not args.run_id:
        parser.error(
            "--run-id is required unless using --self-check."
        )

    try:
        return filter_cohort(
            database_path(),
            args.run_id,
            args.limit,
            args.model,
            args.preview,
        )
    except (
        ClassificationError, ValueError,
        sqlite3.Error, OSError,
    ) as error:
        print(
            f"Filtering stopped: {error}. "
            "Earlier committed results remain saved."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
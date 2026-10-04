"""Validated collection, classification, and filtering records.

Requires Pydantic 2. Run this file for a small local validation demonstration.
The demonstration data is synthetic and is never saved as creator data.
"""

from datetime import datetime, timezone
from decimal import Decimal, localcontext
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationError,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _parse_count(value: object) -> int:
    """Accept integers and API digit strings, rejecting booleans and fractions."""
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    if type(value) is int:
        return value
    raise ValueError("A count must be an integer or an unsigned digit string.")


Count = Annotated[int, Field(ge=0), BeforeValidator(_parse_count)]


class Record(BaseModel):
    """Reject misspelled fields and validate later assignments too."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ChannelRecord(Record):
    channel_id: NonEmptyText
    title: NonEmptyText
    profile_url: HttpUrl
    description: str | None = None
    subscriber_count: Count | None = None
    hidden_subscriber_count: Annotated[bool, Field(strict=True)] | None = None
    country: NonEmptyText | None = None
    uploads_playlist_id: NonEmptyText | None = None
    discovery_query: NonEmptyText
    collected_at: AwareDatetime


class LiveStreamingDetails(Record):
    """Available API live fields; absent values remain unknown."""

    scheduled_start_time: AwareDatetime | None = None
    scheduled_end_time: AwareDatetime | None = None
    actual_start_time: AwareDatetime | None = None
    actual_end_time: AwareDatetime | None = None
    concurrent_viewers: Count | None = None


class VideoRecord(Record):
    video_id: NonEmptyText
    channel_id: NonEmptyText
    video_url: HttpUrl
    title: NonEmptyText
    description: str | None = None
    published_at: AwareDatetime | None = None
    duration: NonEmptyText | None = None  # Preserve the API ISO 8601 duration.
    views: Count | None = None
    likes: Count | None = None
    comments: Count | None = None
    privacy_status: Literal["public", "private", "unlisted"] | None = None
    live_broadcast_content: Literal["none", "upcoming", "live"] | None = None
    live_streaming_details: LiveStreamingDetails | None = None
    default_language: NonEmptyText | None = None
    default_audio_language: NonEmptyText | None = None
    dimension: Literal["2d", "3d"] | None = None
    definition: Literal["hd", "sd"] | None = None
    projection: Literal["rectangular", "360"] | None = None
    collected_at: AwareDatetime


class SizeEligibilityResult(Record):
    """Stores a subscriber-size decision; this is not campaign qualification."""

    channel_id: NonEmptyText
    status: Literal["PASS", "FAIL", "NEEDS_REVIEW"]
    reasons: Annotated[list[NonEmptyText], Field(min_length=1)]
    evaluated_at: AwareDatetime


class CollectionExclusion(Record):
    """Explain why a playlist video was not included in the selected sample."""

    video_id: NonEmptyText
    reason_code: Literal[
        "LIVE_CONTENT", "UPCOMING", "NOT_PUBLIC", "UNAVAILABLE",
        "INVALID_METADATA", "WRONG_CHANNEL", "UNKNOWN_ELIGIBILITY",
    ]
    reason: NonEmptyText
    evidence: list[NonEmptyText] = Field(default_factory=list)


class CollectionResult(Record):
    """One channel's sample for one collection run, ordered newest first.

    COMPLETE means the requested sample was collected without unresolved
    selection gaps. It does not mean the creator passed campaign filtering.
    Regular videos and Shorts share the same chronological selection rule.
    """

    run_id: NonEmptyText
    channel_id: NonEmptyText
    uploads_playlist_id: NonEmptyText | None = None
    requested_video_count: Annotated[int, Field(strict=True, ge=1, le=10)] = 10
    selected_video_ids: list[NonEmptyText] = Field(default_factory=list)
    status: Literal["COMPLETE", "PARTIAL", "FAILED", "SKIPPED"]
    selection_policy: Literal["latest_public_non_live_v1"] = "latest_public_non_live_v1"
    selection_uncertain: Annotated[bool, Field(strict=True)] = False
    exclusions: list[CollectionExclusion] = Field(default_factory=list)
    reasons: list[NonEmptyText] = Field(default_factory=list)
    playlist_pages: Annotated[int, Field(strict=True, ge=0)] = 0
    video_requests: Annotated[int, Field(strict=True, ge=0)] = 0
    started_at: AwareDatetime
    finished_at: AwareDatetime

    @model_validator(mode="after")
    def check_sample(self) -> "CollectionResult":
        identifiers = self.selected_video_ids
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Selected video IDs must be unique.")
        if len(identifiers) > self.requested_video_count:
            raise ValueError("Selected videos must not exceed the requested count.")
        if self.finished_at < self.started_at:
            raise ValueError("Collection cannot finish before it starts.")
        if self.status == "COMPLETE":
            if len(identifiers) != self.requested_video_count or self.selection_uncertain:
                raise ValueError("COMPLETE requires the full sample with no selection uncertainty.")
        elif not self.reasons:
            raise ValueError("An incomplete, failed, or skipped collection needs a reason.")
        if self.status in ("FAILED", "SKIPPED") and identifiers:
            raise ValueError("Use PARTIAL when a failed attempt has saved sample videos.")
        excluded_ids = {item.video_id for item in self.exclusions}
        if excluded_ids.intersection(identifiers):
            raise ValueError("A video cannot be both selected and excluded in one result.")
        return self


Status = Literal["PASS", "FAIL", "NEEDS_REVIEW"]
SampleSize = Annotated[int, Field(strict=True, ge=1, le=10)]
NonnegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _unique_ids(identifiers: list[str]) -> None:
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Video IDs must be unique.")


class TextEvidence(Record):
    """A quotation from the supplied video title or description."""

    source: Literal["title", "description"]
    quote: NonEmptyText


class VideoClassification(Record):
    """LLM interpretation only; Python applies the numerical thresholds."""

    video_id: NonEmptyText
    technology_label: Literal["TECHNOLOGY", "OTHER", "UNCLEAR"]
    technology_reason: NonEmptyText
    technology_evidence: list[TextEvidence]
    relevance_label: Literal["MATCH", "RELATED", "NO_MATCH", "UNCLEAR"]
    relevance_reason: NonEmptyText
    relevance_evidence: list[TextEvidence]

    @model_validator(mode="after")
    def check_evidence(self) -> "VideoClassification":
        if self.technology_label != "UNCLEAR" and not self.technology_evidence:
            raise ValueError("A definite Technology label needs source evidence.")
        if self.relevance_label != "UNCLEAR" and not self.relevance_evidence:
            raise ValueError("A definite relevance label needs source evidence.")
        return self


class ClassificationResponse(Record):
    """The structured object requested from the LLM, without invented metrics."""

    channel_id: NonEmptyText
    channel_niche_reason: NonEmptyText
    channel_description_evidence: list[NonEmptyText]
    videos: Annotated[list[VideoClassification], Field(min_length=1, max_length=10)]

    @model_validator(mode="after")
    def check_ids(self) -> "ClassificationResponse":
        _unique_ids([video.video_id for video in self.videos])
        return self

    def validate_sources(self, channel: ChannelRecord, videos: list[VideoRecord]) -> None:
        """Call with the EXACT records/text supplied to the LLM.

        This checks grounding and completeness, not whether a judgment is right.
        """
        if self.channel_id != channel.channel_id:
            raise ValueError("The LLM returned a different channel ID.")
        for quote in self.channel_description_evidence:
            if quote not in (channel.description or ""):
                raise ValueError("Channel evidence was not found in the supplied description.")
        _unique_ids([video.video_id for video in videos])
        by_id = {video.video_id: video for video in videos}
        if any(video.channel_id != channel.channel_id for video in videos):
            raise ValueError("Input videos must belong to the selected channel.")
        if set(by_id) != {video.video_id for video in self.videos}:
            raise ValueError("The LLM must classify every supplied video exactly once.")
        for judgment in self.videos:
            video = by_id[judgment.video_id]
            for evidence in judgment.technology_evidence + judgment.relevance_evidence:
                text = getattr(video, evidence.source) or ""
                if evidence.quote not in text:
                    raise ValueError("An evidence quote was not found in the supplied text.")


class ClassificationRecord(Record):
    """Saved judgment and identity for reuse; no video metadata snapshots."""

    classification_id: NonEmptyText
    collection_run_id: NonEmptyText
    channel_id: NonEmptyText
    sample_video_ids: Annotated[list[NonEmptyText], Field(min_length=1, max_length=10)]
    input_fingerprint: Fingerprint
    model: NonEmptyText
    prompt_version: NonEmptyText
    criteria_version: NonEmptyText
    response: ClassificationResponse
    classified_at: AwareDatetime

    @model_validator(mode="after")
    def check_links(self) -> "ClassificationRecord":
        _unique_ids(self.sample_video_ids)
        if self.channel_id != self.response.channel_id:
            raise ValueError("Classification channel IDs must agree.")
        if set(self.sample_video_ids) != {video.video_id for video in self.response.videos}:
            raise ValueError("Classification must cover the exact selected sample.")
        return self


class FilteringRules(Record):
    """Store the exact rules used; no language, geography, or activity filters."""

    version: NonEmptyText = "python_course_v1"
    requested_video_count: SampleSize = 10
    technology_min_videos: SampleSize = 5
    relevance_min_videos: SampleSize = 5
    qualifying_relevance_labels: tuple[Literal["MATCH", "RELATED"], ...] = ("MATCH", "RELATED")
    min_subscribers: Count = 5000
    max_subscribers: Count = 100000
    engagement_threshold_percent: NonnegativeDecimal = Decimal("1.4")
    engagement_comparison: Literal["strictly_greater_than"] = "strictly_greater_than"

    @model_validator(mode="after")
    def check_bounds(self) -> "FilteringRules":
        if max(self.technology_min_videos, self.relevance_min_videos) > self.requested_video_count:
            raise ValueError("A content threshold cannot exceed the requested sample.")
        if self.min_subscribers > self.max_subscribers:
            raise ValueError("Subscriber boundaries are reversed.")
        if self.qualifying_relevance_labels != ("MATCH", "RELATED"):
            raise ValueError("Both MATCH and RELATED qualify under the agreed rules.")
        return self


class CriterionResult(Record):
    status: Status
    reasons: Annotated[list[NonEmptyText], Field(min_length=1)]
    qualifying_video_ids: list[NonEmptyText] = Field(default_factory=list)
    unclear_video_ids: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_ids(self) -> "CriterionResult":
        _unique_ids(self.qualifying_video_ids)
        _unique_ids(self.unclear_video_ids)
        if set(self.qualifying_video_ids).intersection(self.unclear_video_ids):
            raise ValueError("A video cannot be qualifying and unclear for the same criterion.")
        return self


class EngagementVideoCounts(Record):
    video_id: NonEmptyText
    likes: Count | None = None
    comments: Count | None = None
    observed_at: AwareDatetime


class EngagementResult(Record):
    """Numerical result, separate from permission to use its data source.

    Rate = 100 * sum(likes + comments) / (requested_video_count * subscribers).
    All requested videos must have known inputs; no silent sample shrinking.
    Views are not part of this subscriber-based denominator.
    """

    formula_version: Literal["mean_interactions_per_subscriber_v1"] = "mean_interactions_per_subscriber_v1"
    requested_video_count: SampleSize = 10
    subscriber_count: Count | None = None
    subscriber_observed_at: AwareDatetime | None = None
    videos: Annotated[list[EngagementVideoCounts], Field(max_length=10)]
    rate_percent: NonnegativeDecimal | None = None
    threshold_percent: NonnegativeDecimal = Decimal("1.4")
    comparison: Literal["strictly_greater_than"] = "strictly_greater_than"
    status: Status
    reasons: Annotated[list[NonEmptyText], Field(min_length=1)]
    # Disclosing a limitation does not establish permission.
    data_use_permission: Literal["NOT_ESTABLISHED", "CONFIRMED"] = "NOT_ESTABLISHED"
    data_use_note: NonEmptyText = "Permission for a derived metric from YouTube API statistics has not been established."

    @model_validator(mode="after")
    def check_calculation(self) -> "EngagementResult":
        _unique_ids([video.video_id for video in self.videos])
        if len(self.videos) > self.requested_video_count:
            raise ValueError("Engagement inputs exceed the requested sample.")
        complete = (
            len(self.videos) == self.requested_video_count
            and self.subscriber_count is not None and self.subscriber_count > 0
            and self.subscriber_observed_at is not None
            and all(video.likes is not None and video.comments is not None for video in self.videos)
        )
        if not complete or self.rate_percent is None:
            if self.rate_percent is not None or self.status != "NEEDS_REVIEW":
                raise ValueError("Missing inputs or an uncalculated rate require NEEDS_REVIEW and no rate.")
            return self
        interactions = sum(video.likes + video.comments for video in self.videos)
        numerator = Decimal(100 * interactions)
        denominator = Decimal(self.requested_video_count * self.subscriber_count)
        with localcontext() as context:
            context.prec = 28
            expected_rate = numerator / denominator
        if self.rate_percent != expected_rate:
            raise ValueError("The saved engagement rate does not match its inputs and formula.")
        # Cross multiplication avoids comparing a rounded percentage at the boundary.
        expected_status = "PASS" if numerator > self.threshold_percent * denominator else "FAIL"
        if self.status != expected_status:
            raise ValueError("Engagement status does not match the strict threshold.")
        return self


class ChannelFilteringResult(Record):
    """Python's combined verdict; only the four agreed criteria participate."""

    filtering_id: NonEmptyText
    collection_run_id: NonEmptyText
    channel_id: NonEmptyText
    sample_video_ids: Annotated[list[NonEmptyText], Field(max_length=10)]
    selection_uncertain: Annotated[bool, Field(strict=True)] = False
    classification_id: NonEmptyText | None = None
    input_fingerprint: Fingerprint
    rules: FilteringRules = Field(default_factory=FilteringRules)
    technology: CriterionResult
    content_relevance: CriterionResult
    subscribers: SizeEligibilityResult
    engagement: EngagementResult
    status: Status
    reasons: Annotated[list[NonEmptyText], Field(min_length=1)]
    evaluated_at: AwareDatetime

    @model_validator(mode="after")
    def check_verdict(self) -> "ChannelFilteringResult":
        _unique_ids(self.sample_video_ids)
        sample = set(self.sample_video_ids)
        if len(sample) > self.rules.requested_video_count:
            raise ValueError("Filtering sample exceeds the configured size.")
        if self.subscribers.channel_id != self.channel_id:
            raise ValueError("Subscriber decision belongs to a different channel.")
        for criterion, minimum in (
            (self.technology, self.rules.technology_min_videos),
            (self.content_relevance, self.rules.relevance_min_videos),
        ):
            if not set(criterion.qualifying_video_ids + criterion.unclear_video_ids).issubset(sample):
                raise ValueError("Content decisions refer to videos outside the selected sample.")
            if criterion.status == "PASS":
                if (len(criterion.qualifying_video_ids) < minimum or self.classification_id is None
                        or len(sample) != self.rules.requested_video_count or self.selection_uncertain):
                    raise ValueError("A content PASS needs sufficient matches and a complete, certain classified sample.")
            if criterion.status == "FAIL":
                possible = (len(criterion.qualifying_video_ids) + len(criterion.unclear_video_ids)
                            + self.rules.requested_video_count - len(sample))
                if possible >= minimum or self.selection_uncertain:
                    raise ValueError("Unresolved content that could meet the threshold requires NEEDS_REVIEW.")
        if set(video.video_id for video in self.engagement.videos) != sample:
            raise ValueError("Engagement must reference the same filtering sample.")
        if (self.engagement.requested_video_count != self.rules.requested_video_count
                or self.engagement.threshold_percent != self.rules.engagement_threshold_percent):
            raise ValueError("Engagement settings must match the filtering rules.")
        statuses = [self.technology.status, self.content_relevance.status,
                    self.subscribers.status, self.engagement.status]
        expected = "FAIL" if "FAIL" in statuses else (
            "NEEDS_REVIEW" if "NEEDS_REVIEW" in statuses else "PASS")
        if self.status != expected:
            raise ValueError("The overall verdict does not match its four criterion results.")
        if self.evaluated_at < self.subscribers.evaluated_at:
            raise ValueError("Overall evaluation predates its subscriber decision.")
        return self


# The older classification/filtering classes above remain readable for saved
# python_course_v1 records. New code must use the title/description models below.


class ChannelNicheClassification(Record):
    """LLM judgment from the channel description ONLY; no copied quotation."""

    label: Literal["TECHNOLOGY", "OTHER", "UNCLEAR"]
    reason: NonEmptyText
    evidence_source: Literal["channel_description", "none"]

    @model_validator(mode="after")
    def check_source(self) -> "ChannelNicheClassification":
        if self.label != "UNCLEAR" and self.evidence_source != "channel_description":
            raise ValueError("A definite niche judgment needs the channel-description source.")
        return self


class TitleRelevanceClassification(Record):
    """LLM interpretation of this video's title ONLY.

    video_id identifies the evidence. Python attaches the stored original title;
    the LLM must not supply or retype a quotation or a video description.
    """

    video_id: NonEmptyText
    relevance_label: Literal["MATCH", "RELATED", "NO_MATCH", "UNCLEAR"]
    relevance_reason: NonEmptyText
    evidence_source: Literal["title"]


class TitleClassificationResponse(Record):
    """One description-based niche judgment and exactly ten title judgments.

    Source validation checks identities and source availability, not semantic
    correctness. Prompts must enforce the separate permitted sources.
    """

    channel_id: NonEmptyText
    niche: ChannelNicheClassification
    videos: Annotated[list[TitleRelevanceClassification], Field(min_length=10, max_length=10)]

    @model_validator(mode="after")
    def check_ids(self) -> "TitleClassificationResponse":
        _unique_ids([video.video_id for video in self.videos])
        return self

    def validate_sources(self, channel: ChannelRecord, videos: list[VideoRecord]) -> None:
        if self.channel_id != channel.channel_id:
            raise ValueError("The LLM returned a different channel ID.")
        if len(videos) != 10:
            raise ValueError("The title-only stage requires exactly ten sample videos.")
        _unique_ids([video.video_id for video in videos])
        if {video.video_id for video in videos} != {video.video_id for video in self.videos}:
            raise ValueError("Every supplied video must be classified exactly once.")
        if any(video.channel_id != channel.channel_id for video in videos):
            raise ValueError("Input videos must belong to the selected channel.")
        if not (channel.description or "").strip():
            if self.niche.label != "UNCLEAR" or self.niche.evidence_source != "none":
                raise ValueError("A missing channel description requires UNCLEAR and no niche evidence.")


class TitleClassificationRecord(Record):
    """Versioned stored record; older quotations-based judgments stay separate."""

    schema_version: Literal["channel_description_titles_v1"] = "channel_description_titles_v1"
    classification_id: NonEmptyText
    collection_run_id: NonEmptyText
    channel_id: NonEmptyText
    sample_video_ids: Annotated[list[NonEmptyText], Field(min_length=10, max_length=10)]
    input_fingerprint: Fingerprint
    model: NonEmptyText
    prompt_version: NonEmptyText
    criteria_version: Literal["python_course_v2"] = "python_course_v2"
    response: TitleClassificationResponse
    classified_at: AwareDatetime

    @model_validator(mode="after")
    def check_links(self) -> "TitleClassificationRecord":
        _unique_ids(self.sample_video_ids)
        if self.channel_id != self.response.channel_id:
            raise ValueError("Classification channel IDs must agree.")
        if set(self.sample_video_ids) != {video.video_id for video in self.response.videos}:
            raise ValueError("Classification must cover the exact selected sample.")
        return self


class TitleFilteringRules(Record):
    """New rules: channel-description niche, five relevant titles out of ten."""

    version: Literal["python_course_v2"] = "python_course_v2"
    technology_source: Literal["channel_description_only"] = "channel_description_only"
    relevance_source: Literal["video_titles_only"] = "video_titles_only"
    requested_video_count: Literal[10] = 10
    relevance_min_videos: SampleSize = 5
    qualifying_relevance_labels: tuple[Literal["MATCH", "RELATED"], ...] = ("MATCH", "RELATED")
    min_subscribers: Count = 5000
    max_subscribers: Count = 100000
    engagement_threshold_percent: NonnegativeDecimal = Decimal("1.4")
    engagement_comparison: Literal["strictly_greater_than"] = "strictly_greater_than"

    @model_validator(mode="after")
    def check_bounds(self) -> "TitleFilteringRules":
        if self.relevance_min_videos > self.requested_video_count:
            raise ValueError("Relevance threshold cannot exceed the requested sample.")
        if self.min_subscribers > self.max_subscribers:
            raise ValueError("Subscriber boundaries are reversed.")
        if self.qualifying_relevance_labels != ("MATCH", "RELATED"):
            raise ValueError("Both MATCH and RELATED qualify under the agreed rules.")
        return self


class ChannelNicheResult(Record):
    """Python's niche criterion; technology is no longer a video-count rule."""

    label: Literal["TECHNOLOGY", "OTHER", "UNCLEAR"] | None = None
    description_available: Annotated[bool, Field(strict=True)]
    evidence_source: Literal["channel_description", "none"]
    status: Status
    reasons: Annotated[list[NonEmptyText], Field(min_length=1)]

    @model_validator(mode="after")
    def check_decision(self) -> "ChannelNicheResult":
        if not self.description_available:
            if self.label not in (None, "UNCLEAR") or self.evidence_source != "none":
                raise ValueError("Missing description cannot support a definite niche judgment.")
        if self.label in ("TECHNOLOGY", "OTHER") and self.evidence_source != "channel_description":
            raise ValueError("A definite niche judgment needs channel-description evidence.")
        if self.label is None and self.evidence_source != "none":
            raise ValueError("An unclassified niche cannot claim source evidence.")
        expected = {"TECHNOLOGY": "PASS", "OTHER": "FAIL"}.get(self.label, "NEEDS_REVIEW")
        if self.status != expected:
            raise ValueError("Niche status does not match the description-based label.")
        return self


class TitleChannelFilteringResult(Record):
    """New Python verdict; niche and title relevance have separate rules."""

    schema_version: Literal["channel_description_titles_v1"] = "channel_description_titles_v1"
    filtering_id: NonEmptyText
    collection_run_id: NonEmptyText
    channel_id: NonEmptyText
    sample_video_ids: Annotated[list[NonEmptyText], Field(max_length=10)]
    selection_uncertain: Annotated[bool, Field(strict=True)] = False
    classification_id: NonEmptyText | None = None
    input_fingerprint: Fingerprint
    rules: TitleFilteringRules = Field(default_factory=TitleFilteringRules)
    technology: ChannelNicheResult
    content_relevance: CriterionResult
    subscribers: SizeEligibilityResult
    engagement: EngagementResult
    status: Status
    reasons: Annotated[list[NonEmptyText], Field(min_length=1)]
    evaluated_at: AwareDatetime

    @model_validator(mode="after")
    def check_verdict(self) -> "TitleChannelFilteringResult":
        _unique_ids(self.sample_video_ids)
        sample = set(self.sample_video_ids)
        if self.subscribers.channel_id != self.channel_id:
            raise ValueError("Subscriber decision belongs to a different channel.")
        relevance = self.content_relevance
        if not set(relevance.qualifying_video_ids + relevance.unclear_video_ids).issubset(sample):
            raise ValueError("Relevance decisions refer to titles outside the selected sample.")
        if self.classification_id is None:
            if self.technology.label is not None or relevance.qualifying_video_ids or relevance.unclear_video_ids:
                raise ValueError("Interpreted niche/title labels require a saved classification.")
            expected_relevance = "NEEDS_REVIEW"
        else:
            minimum = self.rules.relevance_min_videos
            matches = len(relevance.qualifying_video_ids)
            possible = matches + len(relevance.unclear_video_ids) + 10 - len(sample)
            complete = len(sample) == 10 and not self.selection_uncertain
            if matches >= minimum and complete:
                expected_relevance = "PASS"
            elif self.selection_uncertain or possible >= minimum:
                expected_relevance = "NEEDS_REVIEW"
            else:
                expected_relevance = "FAIL"
        if relevance.status != expected_relevance:
            raise ValueError("Relevance status does not match the five-of-ten title rules and uncertainty.")
        if {video.video_id for video in self.engagement.videos} != sample:
            raise ValueError("Engagement must reference the same filtering sample.")
        if (self.engagement.requested_video_count != self.rules.requested_video_count
                or self.engagement.threshold_percent != self.rules.engagement_threshold_percent):
            raise ValueError("Engagement settings must match the filtering rules.")
        statuses = [self.technology.status, relevance.status,
                    self.subscribers.status, self.engagement.status]
        expected = "FAIL" if "FAIL" in statuses else (
            "NEEDS_REVIEW" if "NEEDS_REVIEW" in statuses else "PASS")
        if self.status != expected:
            raise ValueError("The overall verdict does not match its four criterion results.")
        if self.evaluated_at < self.subscribers.evaluated_at:
            raise ValueError("Overall evaluation predates its subscriber decision.")
        return self


def verify_filtering_schemas() -> None:
    """Validate synthetic results only; never classify or score real creators."""
    observed = datetime.now(timezone.utc)
    videos = [VideoRecord(
        video_id=f"synthetic-{number}", channel_id="synthetic-channel",
        video_url=f"https://www.youtube.com/watch?v=synthetic-{number}",
        title="Python project example", description="Learn Python automation.",
        likes=150, comments=20, collected_at=observed,
    ) for number in range(10)]
    channel = ChannelRecord(
        channel_id="synthetic-channel", title="Synthetic check",
        profile_url="https://www.youtube.com/channel/synthetic-channel",
        discovery_query="synthetic", collected_at=observed,
    )
    quote = TextEvidence(source="title", quote="Python project")
    response = ClassificationResponse(
        channel_id=channel.channel_id, channel_niche_reason="Synthetic programming example.",
        channel_description_evidence=[],
        videos=[VideoClassification(
            video_id=video.video_id, technology_label="TECHNOLOGY",
            technology_reason="The supplied title concerns a Python project.", technology_evidence=[quote],
            relevance_label="RELATED", relevance_reason="Python is adjacent to the course.", relevance_evidence=[quote],
        ) for video in videos],
    )
    response.validate_sources(channel, videos)
    record = ClassificationRecord(
        classification_id="synthetic-classification", collection_run_id="synthetic-run",
        channel_id=channel.channel_id, sample_video_ids=[video.video_id for video in videos],
        input_fingerprint="a" * 64, model="synthetic-model", prompt_version="v1",
        criteria_version="python_course_v1", response=response, classified_at=observed,
    )
    engagement = EngagementResult(
        subscriber_count=10000, subscriber_observed_at=observed,
        videos=[EngagementVideoCounts(video_id=video.video_id, likes=video.likes,
                                     comments=video.comments, observed_at=observed) for video in videos],
        rate_percent="1.7", status="PASS", reasons=["Synthetic arithmetic example: 1.7 > 1.4."],
    )
    five_ids = record.sample_video_ids[:5]
    filtering = ChannelFilteringResult(
        filtering_id="synthetic-filter", collection_run_id=record.collection_run_id,
        channel_id=channel.channel_id, sample_video_ids=record.sample_video_ids,
        classification_id=record.classification_id, input_fingerprint="b" * 64,
        technology=CriterionResult(status="PASS", reasons=["Synthetic five-of-ten check."], qualifying_video_ids=five_ids),
        content_relevance=CriterionResult(status="PASS", reasons=["Synthetic RELATED labels qualify."], qualifying_video_ids=five_ids),
        subscribers=SizeEligibilityResult(channel_id=channel.channel_id, status="PASS",
                                         reasons=["Synthetic size check."], evaluated_at=observed),
        engagement=engagement, status="PASS", reasons=["All four numerical/content checks pass in this synthetic example."],
        evaluated_at=observed,
    )
    for model in (record, filtering):
        if type(model).model_validate_json(model.model_dump_json()) != model:
            raise RuntimeError("Filtering record serialization failed.")

    def reject(model, values):
        try:
            model.model_validate(values)
        except ValidationError:
            return
        raise RuntimeError("An inconsistent filtering record was accepted.")

    reject(ClassificationResponse, {**response.model_dump(), "videos": [response.videos[0].model_dump()] * 2})
    reject(VideoClassification, {**response.videos[0].model_dump(), "relevance_label": "LIKELY"})
    reject(ClassificationRecord, {**record.model_dump(), "input_fingerprint": "not-a-hash"})
    try:
        response.validate_sources(channel, videos[:-1])
    except ValueError:
        pass
    else:
        raise RuntimeError("Incomplete source coverage was accepted.")
    invented = response.model_dump()
    invented["videos"][0]["relevance_evidence"][0]["quote"] = "invented source text"
    try:
        ClassificationResponse.model_validate(invented).validate_sources(channel, videos)
    except ValueError:
        pass
    else:
        raise RuntimeError("Invented evidence was accepted.")
    missing = engagement.model_dump()
    missing["videos"][0]["likes"] = None
    reject(EngagementResult, missing)
    EngagementResult.model_validate({**missing, "rate_percent": None, "status": "NEEDS_REVIEW"})
    boundary = engagement.model_dump()
    for counts in boundary["videos"]:
        counts.update(likes=120, comments=20)
    boundary.update(rate_percent="1.4", status="FAIL")
    EngagementResult.model_validate(boundary)
    reject(EngagementResult, {**boundary, "status": "PASS"})
    reject(EngagementResult, {**engagement.model_dump(), "rate_percent": "NaN"})
    wrong_verdict = filtering.model_dump()
    wrong_verdict["technology"]["status"] = "NEEDS_REVIEW"
    reject(ChannelFilteringResult, wrong_verdict)
    ChannelFilteringResult.model_validate({**wrong_verdict, "status": "NEEDS_REVIEW"})
    wrong_verdict["subscribers"]["status"] = "FAIL"
    ChannelFilteringResult.model_validate({**wrong_verdict, "status": "FAIL"})
    print("Legacy filtering schemas OK: older records remain readable; prior rules are historical only.")
    print("Synthetic validation only; no LLM calls, real-data rates, API requests, or database writes.")


def verify_title_schemas() -> None:
    """Validate new rules without classifying real creators or writing files."""
    observed = datetime.now(timezone.utc)
    channel = ChannelRecord(
        channel_id="synthetic-channel", title="Synthetic check",
        description="Python software development tutorials.",
        profile_url="https://www.youtube.com/channel/synthetic-channel",
        discovery_query="synthetic", collected_at=observed,
    )
    videos = [VideoRecord(
        video_id=f"synthetic-{number}", channel_id=channel.channel_id,
        video_url=f"https://www.youtube.com/watch?v=synthetic-{number}",
        title="Python project example", description="Unused video description.",
        collected_at=observed,
    ) for number in range(10)]
    response = TitleClassificationResponse(
        channel_id=channel.channel_id,
        niche=ChannelNicheClassification(label="TECHNOLOGY",
            reason="The channel description concerns software development.",
            evidence_source="channel_description"),
        videos=[TitleRelevanceClassification(video_id=video.video_id,
            relevance_label="RELATED" if number < 5 else "NO_MATCH",
            relevance_reason="Synthetic title interpretation only.", evidence_source="title")
            for number, video in enumerate(videos)],
    )
    response.validate_sources(channel, videos)
    # Stored video descriptions do not participate in the new source checks.
    response.validate_sources(channel, [VideoRecord.model_validate({
        **video.model_dump(), "description": None,
    }) for video in videos])
    record = TitleClassificationRecord(
        classification_id="synthetic-title-classification", collection_run_id="synthetic-run",
        channel_id=channel.channel_id, sample_video_ids=[video.video_id for video in videos],
        input_fingerprint="c" * 64, model="synthetic-model", prompt_version="synthetic-title-v1",
        response=response, classified_at=observed,
    )
    result = TitleChannelFilteringResult(
        filtering_id="synthetic-title-filter", collection_run_id=record.collection_run_id,
        channel_id=channel.channel_id, sample_video_ids=record.sample_video_ids,
        classification_id=record.classification_id, input_fingerprint="d" * 64,
        technology=ChannelNicheResult(label="TECHNOLOGY", description_available=True,
            evidence_source="channel_description", status="PASS", reasons=["Synthetic niche decision."]),
        content_relevance=CriterionResult(status="PASS", reasons=["Five related titles qualify."],
            qualifying_video_ids=record.sample_video_ids[:5]),
        subscribers=SizeEligibilityResult(channel_id=channel.channel_id, status="PASS",
            reasons=["Synthetic size decision."], evaluated_at=observed),
        engagement=EngagementResult(subscriber_count=10000, subscriber_observed_at=observed,
            videos=[EngagementVideoCounts(video_id=video.video_id, likes=150, comments=20,
                observed_at=observed) for video in videos], rate_percent="1.7", status="PASS",
            reasons=["Synthetic arithmetic only."]),
        status="PASS", reasons=["Synthetic combined verdict."], evaluated_at=observed,
    )
    for item in (response, record, result):
        if type(item).model_validate_json(item.model_dump_json()) != item:
            raise RuntimeError("New classification/filtering serialization failed.")

    def reject(model, values):
        try:
            model.model_validate(values)
        except ValidationError:
            return
        raise RuntimeError("An inconsistent new-schema example was accepted.")

    reject(TitleRelevanceClassification, {**response.videos[0].model_dump(), "evidence_source": "description"})
    reject(TitleRelevanceClassification, {**response.videos[0].model_dump(), "quote": "invented"})
    reject(TitleRelevanceClassification, {**response.videos[0].model_dump(), "technology_label": "TECHNOLOGY"})
    reject(TitleClassificationResponse, {**response.model_dump(), "videos": [response.videos[0].model_dump()] * 10})
    reject(TitleClassificationResponse, {**response.model_dump(), "videos": response.model_dump()["videos"][:-1]})
    reject(TitleClassificationRecord, {**record.model_dump(), "criteria_version": "python_course_v1"})
    reject(TitleFilteringRules, {"technology_min_videos": 5})
    reject(ChannelNicheClassification, {**response.niche.model_dump(), "evidence_source": "none"})
    empty_channel = ChannelRecord.model_validate({**channel.model_dump(), "description": "   "})
    try:
        response.validate_sources(empty_channel, videos)
    except ValueError:
        pass
    else:
        raise RuntimeError("Missing description was accepted as a definite niche.")
    unclear_data = response.model_dump()
    unclear_data["niche"].update(label="UNCLEAR", evidence_source="none", reason="Description is missing.")
    TitleClassificationResponse.model_validate(unclear_data).validate_sources(empty_channel, videos)
    wrong_ids = response.model_dump()
    wrong_ids["videos"][0]["video_id"] = "outside-sample"
    try:
        TitleClassificationResponse.model_validate(wrong_ids).validate_sources(channel, videos)
    except ValueError:
        pass
    else:
        raise RuntimeError("A video ID outside the supplied sample was accepted.")
    for matches, uncertain, expected in ((5, 0, "PASS"), (4, 0, "FAIL"),
            (4, 1, "NEEDS_REVIEW"), (3, 1, "FAIL"), (5, 1, "PASS")):
        data = result.model_dump()
        data["content_relevance"].update(
            status=expected, qualifying_video_ids=record.sample_video_ids[:matches],
            unclear_video_ids=record.sample_video_ids[matches:matches + uncertain],
        )
        data["status"] = expected
        TitleChannelFilteringResult.model_validate(data)
        wrong = {**data, "status": "FAIL" if expected != "FAIL" else "PASS"}
        reject(TitleChannelFilteringResult, wrong)
    data = result.model_dump()
    data["technology"].update(label="OTHER", status="FAIL")
    data["status"] = "FAIL"
    TitleChannelFilteringResult.model_validate(data)
    data = result.model_dump()
    data["technology"].update(label="UNCLEAR", description_available=False,
        evidence_source="none", status="NEEDS_REVIEW")
    data["status"] = "NEEDS_REVIEW"
    TitleChannelFilteringResult.model_validate(data)
    reject(TitleChannelFilteringResult, {**data, "status": "PASS"})
    print("Title-only schemas OK: channel-description niche, ten title IDs, no quotations, missing-description review, five-of-ten relevance, and overall verdicts.")


def main() -> None:
    """Exercise validation using clearly labelled synthetic inputs."""
    observed_at = datetime.now(timezone.utc)
    channel = ChannelRecord(
        channel_id="synthetic-channel",
        title="Synthetic validation example",
        profile_url="https://www.youtube.com/channel/synthetic-channel",
        subscriber_count="5000",
        hidden_subscriber_count=False,
        discovery_query="Python projects for beginners",
        collected_at=observed_at,
    )
    video = VideoRecord(
        video_id="synthetic-video",
        channel_id=channel.channel_id,
        video_url="https://www.youtube.com/watch?v=synthetic-video",
        title="Synthetic validation example",
        views="0",
        collected_at=observed_at,
    )
    result = SizeEligibilityResult(
        channel_id=channel.channel_id,
        status="NEEDS_REVIEW",
        reasons=["Synthetic example only; no actual filtering was performed."],
        evaluated_at=observed_at,
    )
    channel_round_trip = ChannelRecord.model_validate_json(channel.model_dump_json())
    if channel_round_trip != channel or video.likes is not None or video.views != 0:
        raise RuntimeError("Serialization or missing-count validation failed.")

    for invalid_fields in (
        {"views": -1},
        {"views": True},
        {"views": 1.5},
        {"collected_at": datetime(2026, 1, 1)},
        {"unexpected_field": "not allowed"},
    ):
        try:
            VideoRecord.model_validate({**video.model_dump(), **invalid_fields})
        except ValidationError:
            continue
        raise RuntimeError("An invalid example was accepted.")

    print("Schemas OK (synthetic validation only; no API calls or files written).")
    print(f"Subscriber digit string converted to integer: {channel.subscriber_count}")
    print(f"Missing likes preserved: {video.likes}; reported zero views preserved: {video.views}")
    print("Rejected negative counts, booleans, fractional counts, naive dates, and extra fields.")
    print(f"Size status example: {result.status} (not a real creator decision).")

    collection = CollectionResult(
        run_id="synthetic-run",
        channel_id=channel.channel_id,
        selected_video_ids=[video.video_id],
        status="PARTIAL",
        reasons=["Synthetic example: only one sample video is available."],
        exclusions=[CollectionExclusion(
            video_id="synthetic-live-video",
            reason_code="LIVE_CONTENT",
            reason="Synthetic example of an excluded archived live broadcast.",
            evidence=["liveStreamingDetails.actualStartTime is present"],
        )],
        started_at=observed_at,
        finished_at=observed_at,
    )
    if CollectionResult.model_validate_json(collection.model_dump_json()) != collection:
        raise RuntimeError("Collection result serialization failed.")
    for changes in (
        {"selected_video_ids": [video.video_id, video.video_id]},
        {"status": "COMPLETE"},
        {"reasons": []},
        {"status": "SKIPPED"},
        {"selected_video_ids": ["synthetic-live-video"]},
        {"finished_at": "2000-01-01T00:00:00Z"},
    ):
        try:
            CollectionResult.model_validate({**collection.model_dump(), **changes})
        except ValidationError:
            continue
        raise RuntimeError("An inconsistent collection example was accepted.")
    complete_data = collection.model_dump()
    complete_data.update(
        status="COMPLETE", selected_video_ids=[f"synthetic-{number}" for number in range(10)],
        reasons=[], exclusions=[],
    )
    CollectionResult.model_validate(complete_data)
    try:
        CollectionResult.model_validate({**complete_data, "selection_uncertain": True})
    except ValidationError:
        pass
    else:
        raise RuntimeError("An uncertain sample was incorrectly accepted as COMPLETE.")
    print("Collection schemas OK: sample IDs, exclusions, statuses, reasons, and timestamps.")
    verify_filtering_schemas()
    verify_title_schemas()


if __name__ == "__main__":
    main()

"""Human review of saved drafts. Local settings; no sending or email checks."""
from config import database_path, load_environment
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory


def connect(path, writable=False):
    path = Path(path).resolve()
    connection = sqlite3.connect(path.as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def load_profiles(connection, run_id, limit=None):
    """Read current PASS enrichments and their latest exact saved draft pair."""
    rows = connection.execute("""
        SELECT f.* FROM filtering_results f
        WHERE f.collection_run_id=? AND NOT EXISTS (
          SELECT 1 FROM filtering_results newer
          WHERE newer.collection_run_id=f.collection_run_id AND newer.channel_id=f.channel_id
            AND (newer.evaluated_at, newer.filtering_id) > (f.evaluated_at, f.filtering_id))
          AND f.status='PASS' ORDER BY f.channel_id
    """, (run_id,)).fetchall()
    profiles = []
    for filtering in rows:
        saved = connection.execute("SELECT * FROM profile_enrichments WHERE filtering_id=? ORDER BY enriched_at DESC, enrichment_id DESC LIMIT 1", (filtering["filtering_id"],)).fetchone()
        if saved is None:
            continue
        profile = json.loads(saved["record_json"])
        for key, expected in (("enrichment_id", saved["enrichment_id"]), ("filtering_id", filtering["filtering_id"]), ("channel_id", filtering["channel_id"]), ("collection_run_id", run_id), ("email_status", saved["email_status"])):
            if profile.get(key) != expected:
                raise ValueError("Saved enrichment links or status are inconsistent.")
        draft_row = connection.execute("SELECT * FROM personalized_messages WHERE enrichment_id=? ORDER BY generated_at DESC, message_id DESC LIMIT 1", (saved["enrichment_id"],)).fetchone()
        if draft_row is None:
            continue
        draft = json.loads(draft_row["record_json"])
        for key, expected in (("message_id", draft_row["message_id"]), ("enrichment_id", saved["enrichment_id"]), ("filtering_id", filtering["filtering_id"]), ("channel_id", filtering["channel_id"]), ("collection_run_id", run_id)):
            if draft.get(key) != expected or draft_row[key] != expected:
                raise ValueError("Saved draft links are inconsistent.")
        response = draft.get("response", {})
        if not all(isinstance(response.get(key), str) and response[key].strip() for key in ("email_subject", "email_body", "instagram_dm")):
            raise ValueError("Saved draft text is incomplete.")
        profiles.append((profile, draft))
    return profiles if limit is None else profiles[:limit]


def fingerprint(profile, draft, medium):
    response = draft["response"]
    bound = {key: draft[key] for key in ("message_id", "enrichment_id", "filtering_id", "collection_run_id", "channel_id")}
    bound["medium"] = medium
    if medium == "email":
        bound.update(subject=response["email_subject"], body=response["email_body"], recipient=profile.get("contact_email"), email_status=profile["email_status"])
    else:
        bound.update(body=response["instagram_dm"], recipients=profile.get("instagram_urls", []))
    return hashlib.sha256(json.dumps(bound, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def decision(connection, profile, draft, medium):
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='draft_reviews'").fetchone():
        return None
    return connection.execute("SELECT * FROM draft_reviews WHERE message_id=? AND medium=? AND fingerprint=? ORDER BY review_id DESC LIMIT 1", (draft["message_id"], medium, fingerprint(profile, draft, medium))).fetchone()


def record_review(path, run_id, message_id, medium, verdict, reviewer, reason=""):
    if medium not in ("email", "dm") or verdict not in ("APPROVED", "REJECTED"):
        raise ValueError("Invalid review decision.")
    reviewer, reason = reviewer.strip(), reason.strip()
    if not reviewer or (verdict == "REJECTED" and not reason):
        raise ValueError("A reviewer is required; rejection also requires a reason.")
    with closing(connect(path, True)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        matches = [(p, d) for p, d in load_profiles(connection, run_id) if d["message_id"] == message_id]
        if len(matches) != 1:
            raise ValueError("Draft is not the current saved draft of an enriched PASS profile in this run.")
        profile, draft = matches[0]
        connection.execute("""CREATE TABLE IF NOT EXISTS draft_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT NOT NULL REFERENCES personalized_messages(message_id),
            medium TEXT NOT NULL CHECK(medium IN ('email','dm')),
            fingerprint TEXT NOT NULL,
            verdict TEXT NOT NULL CHECK(verdict IN ('APPROVED','REJECTED')),
            reviewer TEXT NOT NULL, reason TEXT NOT NULL, reviewed_at TEXT NOT NULL
        )""")
        previous = decision(connection, profile, draft, medium)
        if previous and (previous["verdict"], previous["reviewer"], previous["reason"]) == (verdict, reviewer, reason):
            return previous["review_id"]
        cursor = connection.execute("INSERT INTO draft_reviews(message_id,medium,fingerprint,verdict,reviewer,reason,reviewed_at) VALUES(?,?,?,?,?,?,?)", (message_id, medium, fingerprint(profile, draft, medium), verdict, reviewer, reason, datetime.now(timezone.utc).isoformat()))
        return cursor.lastrowid


def get_approved_emails(path, run_id):
    """Read-only handoff for the future sending layer; no deliverability claim."""
    ready = []
    with closing(connect(path)) as connection:
        for profile, draft in load_profiles(connection, run_id):
            reviewed = decision(connection, profile, draft, "email")
            if reviewed and reviewed["verdict"] == "APPROVED" and profile["email_status"] == "FOUND" and profile.get("contact_email"):
                ready.append(dict(message_id=draft["message_id"], channel_id=draft["channel_id"], collection_run_id=run_id, enrichment_id=profile["enrichment_id"], recipient=profile["contact_email"], subject=draft["response"]["email_subject"], body=draft["response"]["email_body"], review_id=reviewed["review_id"], review_fingerprint=reviewed["fingerprint"], reviewed_by=reviewed["reviewer"], reviewed_at=reviewed["reviewed_at"], email_deliverability=profile.get("email_deliverability", "NOT_VERIFIED")))
    return ready


def display(path, run_id, limit):
    with closing(connect(path)) as connection:
        profiles = load_profiles(connection, run_id, limit)
        for profile, draft in profiles:
            print(f"\n{profile['influencer_name']} | {draft['channel_id']}\nMessage ID: {draft['message_id']}")
            print(f"Saved contact: {profile.get('contact_email') or profile['email_status']}")
            for medium in ("email", "dm"):
                reviewed = decision(connection, profile, draft, medium)
                print(f"{medium.upper()} review: {reviewed['verdict'] if reviewed else 'PENDING_REVIEW'}")
            response = draft["response"]
            print(f"Subject: {response['email_subject']}\nEmail:\n{response['email_body']}\nInstagram DM:\n{response['instagram_dm']}")
            print("Email contact available." if profile["email_status"] == "FOUND" and profile.get("contact_email") else "Email handoff blocked: no saved selected FOUND contact.")
        print(f"\nSaved draft pairs shown: {len(profiles)}. No messages sent; no email revalidation.")


def self_check():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "check.db"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.executescript("""
                CREATE TABLE filtering_results(filtering_id TEXT PRIMARY KEY,collection_run_id TEXT,channel_id TEXT,status TEXT,evaluated_at TEXT);
                CREATE TABLE profile_enrichments(enrichment_id TEXT PRIMARY KEY,filtering_id TEXT,channel_id TEXT,email_status TEXT,enriched_at TEXT,record_json TEXT);
                CREATE TABLE personalized_messages(message_id TEXT PRIMARY KEY,enrichment_id TEXT,filtering_id TEXT,collection_run_id TEXT,channel_id TEXT,generated_at TEXT,record_json TEXT);
            """)
            for number, status in enumerate(("FOUND", "NOT_FOUND")):
                suffix = str(number)
                profile = dict(enrichment_id="e"+suffix, filtering_id="f"+suffix, collection_run_id="check", channel_id=suffix, influencer_name="Synthetic", email_status=status, contact_email="observed@example.test" if status == "FOUND" else None, instagram_urls=[])
                draft = dict(message_id="m"+suffix, enrichment_id="e"+suffix, filtering_id="f"+suffix, collection_run_id="check", channel_id=suffix, response=dict(email_subject="Synthetic", email_body="Synthetic saved email", instagram_dm="Synthetic saved DM"))
                connection.execute("INSERT INTO filtering_results VALUES(?,?,?,?,?)", ("f"+suffix,"check",suffix,"PASS","2026-01-01"))
                connection.execute("INSERT INTO profile_enrichments VALUES(?,?,?,?,?,?)", ("e"+suffix,"f"+suffix,suffix,status,"2026-01-01",json.dumps(profile)))
                connection.execute("INSERT INTO personalized_messages VALUES(?,?,?,?,?,?,?)", ("m"+suffix,"e"+suffix,"f"+suffix,"check",suffix,"2026-01-01",json.dumps(draft)))
        assert get_approved_emails(path,"check") == []
        with closing(connect(path)) as connection:
            assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name='draft_reviews'").fetchone()
        record_review(path,"check","m0","dm","APPROVED","Synthetic reviewer")
        assert get_approved_emails(path,"check") == []
        first = record_review(path,"check","m0","email","APPROVED","Synthetic reviewer")
        assert first == record_review(path,"check","m0","email","APPROVED","Synthetic reviewer")
        record_review(path,"check","m1","email","APPROVED","Synthetic reviewer")
        assert len(get_approved_emails(path,"check")) == 1
        record_review(path,"check","m0","email","REJECTED","Synthetic reviewer","Revoked")
        assert get_approved_emails(path,"check") == []
        record_review(path,"check","m0","email","APPROVED","Synthetic reviewer")
        with closing(sqlite3.connect(path)) as connection, connection:
            row = connection.execute("SELECT record_json FROM personalized_messages WHERE message_id='m0'").fetchone()
            changed = json.loads(row[0]); changed["response"]["email_body"] += " changed"
            connection.execute("UPDATE personalized_messages SET record_json=? WHERE message_id='m0'", (json.dumps(changed),))
        assert get_approved_emails(path,"check") == []
        record_review(path,"check","m0","email","APPROVED","Synthetic reviewer")
        with closing(sqlite3.connect(path)) as connection, connection:
            row = connection.execute("SELECT record_json FROM profile_enrichments WHERE enrichment_id='e0'").fetchone()
            changed = json.loads(row[0]); changed["contact_email"] = "different@example.test"
            connection.execute("UPDATE profile_enrichments SET record_json=? WHERE enrichment_id='e0'", (json.dumps(changed),))
        assert get_approved_emails(path,"check") == []
        for args in (("m0","email","REJECTED","Reviewer",""),("absent","email","APPROVED","Reviewer","")):
            try:
                record_review(path,"check",*args)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid review accepted")
    print("Review checks OK: read-only preview, separate media, approvals, rejection, idempotence, changed text/recipient and missing contacts.")
    print("Synthetic temporary database only; no API calls, email revalidation or messages sent.")


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--ready", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--database", default=str(database_path()))
    parser.add_argument("--limit", type=int, default=3)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--approve", metavar="MESSAGE_ID")
    actions.add_argument("--reject", metavar="MESSAGE_ID")
    parser.add_argument("--medium", choices=("email","dm"), default="email")
    parser.add_argument("--reviewer")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return 0
    if not args.run_id or args.limit < 1:
        parser.error("--run-id and a positive --limit are required")
    if (args.approve or args.reject) and (args.preview or args.ready):
        parser.error("Review decisions cannot be combined with --preview or --ready")
    try:
        if args.approve or args.reject:
            if not args.reviewer:
                parser.error("An explicit --reviewer is required")
            review_id = record_review(args.database,args.run_id,args.approve or args.reject,args.medium,"APPROVED" if args.approve else "REJECTED",args.reviewer,args.reason)
            print(f"Review saved: {review_id}. No messages sent.")
        elif args.ready:
            print(json.dumps(get_approved_emails(args.database,args.run_id),indent=2,ensure_ascii=False))
            print("Read-only handoff. No messages sent; sending layer must handle duplicate prevention and logging.")
        else:
            display(args.database,args.run_id,args.limit)
        return 0
    except (sqlite3.Error, ValueError, KeyError, TypeError, OSError) as error:
        print(f"Review failed: {error}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())

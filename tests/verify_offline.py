"""Run existing synthetic checks and focused integration checks without network.

python tests/verify_offline.py
python tests/verify_offline.py --database data/outreach.db --run-id RUN_ID
The optional database is copied before any operation. No original writes.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocked(*args, **kwargs):
    raise AssertionError('Offline check attempted network access')


def configuration():
    from config import load_settings
    from config import database_path, load_environment
    with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
        local = Path(directory) / '.env'
        local.write_text('YOUTUBE_API_KEY=synthetic\nGROQ_API_KEY="synthetic groq"\n'
                         'GEMINI_API_KEY=synthetic\nDISCOVERY_TARGET=51\n'
                         'DATABASE_PATH=data/synthetic.db\n', encoding='utf-8')
        os.environ['DISCOVERY_TARGET'] = '50'
        load_environment(local)
        settings = load_settings(dict(os.environ))
        assert settings.discovery_target == 50
        assert settings.database_path == database_path() == ROOT / 'data' / 'synthetic.db'
        assert os.environ['GROQ_API_KEY'] == 'synthetic groq'
        os.environ['GEMINI_API_KEY'] = ''
        load_environment(local)
        assert os.environ['GEMINI_API_KEY'] == ''
        assert database_path({}) == ROOT / 'data' / 'outreach.db'
        assert database_path({'DATABASE_PATH': '  '}) == database_path({})
        absolute = Path(directory) / 'chosen.db'
        assert database_path({'DATABASE_PATH': str(absolute)}) == absolute
        # Use a disposable project directory to test the real config entry point
        # and its default .env discovery, without creating a deliverable secret file.
        project = Path(directory) / 'project'
        project.mkdir()
        for name in ('config.py',):
            shutil.copy2(ROOT / name, project / name)
        (project / '.env').write_text('YOUTUBE_API_KEY=synthetic-from-file\nDISCOVERY_TARGET=51\n'
                                    'DATABASE_PATH=data/from-file.db\n', encoding='utf-8')
        process_env = dict(os.environ)
        for name in ('YOUTUBE_API_KEY', 'DATABASE_PATH'):
            process_env.pop(name, None)
        result = subprocess.run([sys.executable, str(project / 'config.py')],
                                env=process_env, capture_output=True, text=True, check=True)
        assert 'synthetic-from-file' not in result.stdout
        summary = json.loads(result.stdout.split('\n', 1)[1])
        assert summary['discovery_target'] == 50
        assert Path(summary['database_path']) == project / 'data' / 'from-file.db'


def reservations():
    import sending
    import gmail_demo
    profile = dict(influencer_name='Synthetic', email_status='FOUND',
                   contact_email='creator@example.test', instagram_urls=[])
    draft = dict(channel_id='c', campaign_id='test', message_id='m',
                 response=dict(email_subject='Exact subject', email_body='Exact body', instagram_dm='Exact DM'))
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'test.db'
        with contextlib.closing(sqlite3.connect(path)) as connection, connection:
            connection.execute('CREATE TABLE personalized_messages(message_id TEXT,campaign_id TEXT)')
            connection.executemany('INSERT INTO personalized_messages VALUES(?,?)', [('m', 'test'), ('new', 'test')])
        with patch.object(sending, 'load_profiles', return_value=[(profile, draft)]), \
             patch.object(sending, 'decision', return_value={'verdict': 'APPROVED', 'review_id': 1}), \
             patch.object(sending, 'fingerprint', return_value='exact'):
            assert sending.run(path, 'first', simulate=True)[0]['status'] == 'SIMULATED'
            saved = sending.tracker(path, 'first')[0]
            draft['message_id'] = 'new'
            assert sending.run(path, 'new-run', simulate=True)[0]['status'] == 'SKIPPED_DUPLICATE'
            assert sending.tracker(path, 'first') == [saved]
        approved = dict(channel_id='c', message_id='m', recipient='creator@example.test',
                        subject='Exact subject', body='Exact body', review_id=1)
        service = MagicMock()
        execute = service.users.return_value.messages.return_value.send.return_value.execute
        def accepted(**kwargs):
            assert kwargs == {'num_retries': 0}
            # A separate reader sees the committed reservation BEFORE acceptance.
            with contextlib.closing(sqlite3.connect(path)) as connection:
                status = connection.execute("SELECT status FROM gmail_test_log WHERE test_recipient='tester@example.test'").fetchone()[0]
            assert status == 'IN_PROGRESS'
            return {'id': 'fake-provider-id'}
        execute.side_effect = accepted
        with patch.object(gmail_demo, 'approved', return_value=approved):
            result = gmail_demo.send_test(path, 'first', 'm', 'tester@example.test', service)
            assert result['status'] == 'SENT' and result['gmail_message_id'] == 'fake-provider-id'
            approved['message_id'] = 'new'
            assert gmail_demo.send_test(path, 'new-run', 'new', 'TESTER@example.test', service)['status'] == 'SKIPPED_DUPLICATE_OR_UNCERTAIN'
            assert execute.call_count == 1
            execute.side_effect = TimeoutError()
            assert gmail_demo.send_test(path, 'new-run', 'new', 'uncertain@example.test', service)['status'] == 'UNKNOWN'
            assert gmail_demo.send_test(path, 'new-run', 'new', 'uncertain@example.test', service)['status'] == 'SKIPPED_DUPLICATE_OR_UNCERTAIN'
            with contextlib.closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE gmail_test_log SET status='IN_PROGRESS' WHERE test_recipient='uncertain@example.test'")
            assert gmail_demo.send_test(path, 'new-run', 'new', 'uncertain@example.test', service)['status'] == 'SKIPPED_DUPLICATE_OR_UNCERTAIN'
        assert sending.tracker(path, 'first') == [saved]


def compatibility(source, run_id):
    import database
    import schemas
    import classification
    import filtering
    import enrichment
    import personalization
    import review
    import sending
    import gmail_demo
    import export_results
    before = digest(source)
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'copy.db'
        shutil.copy2(source, path)
        copy_before = digest(path)
        with contextlib.closing(review.connect(path)) as connection:
            assert connection.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            assert not connection.execute('PRAGMA foreign_key_check').fetchall()
            readers = {
                'influencers': schemas.ChannelRecord.model_validate_json,
                'videos': schemas.VideoRecord.model_validate_json,
                'collection_results': schemas.CollectionResult.model_validate_json,
                'classifications': database._parse_classification_record,
                'filtering_results': database._parse_filtering_record,
                'profile_enrichments': enrichment.EnrichedProfile.model_validate_json,
                'personalized_messages': personalization.MessageRecord.model_validate_json,
            }
            counts = {}
            for table, reader in readers.items():
                rows = connection.execute(f'SELECT record_json FROM {table}').fetchall()
                for row in rows:
                    reader(row[0])
                counts[table] = len(rows)
            total, items = classification.read_cohort(path, run_id, 5000)
            assert total == 50 and len(items) == 50
            for collection, channel, videos, problem in items:
                assert problem is None
                assert len(videos) == 10
                assert [v.video_id for v in videos] == collection.selected_video_ids
                assert all(v.channel_id == channel.channel_id for v in videos)
            data, profiles, messages, simulation, gmail = export_results.collect(connection, run_id)
            pairs = review.load_profiles(connection, run_id)
            assert len(profiles) == len(messages) == len(pairs) == 3
            for profile, draft in pairs:
                assert any(m['draft'] == draft and m['profile'] == profile for m in messages)
            historical_reviews = [tuple(row) for row in connection.execute('SELECT * FROM draft_reviews')]
            historical_sim = sending.tracker(path, run_id)
            historical_gmail = gmail_demo.test_log(path)
        # Read-only CLI entry points compose through the same DATABASE_PATH.
        with patch.dict(os.environ, {'DATABASE_PATH': str(path)}, clear=True):
            for module in (classification, filtering, enrichment, personalization, review, sending):
                with patch.object(sys, 'argv', [module.__name__ + '.py', '--preview', '--run-id', run_id, '--limit', '50']):
                    assert module.main() == 0, module.__name__
            ready = review.get_approved_emails(path, run_id)
            assert len(ready) == 1
            with patch.object(sys, 'argv', ['gmail_demo.py', '--preview', '--run-id', run_id,
                                          '--message-id', ready[0]['message_id'], '--test-recipient', 'tester@example.test']):
                assert gmail_demo.main() == 0
        summary = export_results.export_results(path, run_id, Path(directory) / 'exports')
        assert summary['filtering_status_counts'] == {'FAIL': 44, 'PASS': 3, 'NEEDS_REVIEW': 3}
        assert summary['cohort_influencers'] == 50 and summary['personalized_draft_pairs'] == 3
        with (Path(directory) / 'exports' / 'influencers.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 50
        assert {r['Channel_ID'] for r in rows} == {item[0].channel_id for item in items}
        assert sum(r['Email_Status'] == 'NOT_ENRICHED' for r in rows) == 47
        assert sum(r['Email_Status'] == 'NOT_FOUND' for r in rows) == 2
        exported_text = (Path(directory) / 'exports' / 'personalized_messages.md').read_text(encoding='utf-8')
        for _, draft in pairs:
            for key in ('email_body', 'instagram_dm'):
                assert '\n'.join('    ' + line for line in draft['response'][key].split('\n')) in exported_text
        assert digest(path) == copy_before
        with contextlib.closing(review.connect(path)) as connection:
            assert historical_reviews == [tuple(row) for row in connection.execute('SELECT * FROM draft_reviews')]
        assert historical_sim == sending.tracker(path, run_id)
        assert historical_gmail == gmail_demo.test_log(path)
        # Version-4 initialization on the copy preserves schema and all history.
        with database.open_database(path):
            pass
        assert digest(path) == copy_before
    assert digest(source) == before
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if bool(args.database) != bool(args.run_id):
        parser.error('--database and --run-id must be supplied together')
    checks = []
    # Imports themselves must not open a database, load .env or use network.
    names = ('config', 'schemas', 'database', 'discovery', 'collection', 'classification',
             'filtering', 'enrichment', 'personalization', 'review', 'sending', 'gmail_demo', 'export_results')
    with patch.object(sqlite3, 'connect', side_effect=blocked), \
         patch.object(socket.socket, 'connect', blocked), \
         patch('dotenv.load_dotenv', side_effect=blocked):
        for name in names:
            importlib.import_module(name)
    checks.append(('import boundaries', lambda: None))
    checks.append(('configuration', configuration))
    for name in names:
        module = importlib.import_module(name)
        if hasattr(module, 'self_check'):
            checks.append((name, module.self_check))
    database = importlib.import_module('database')
    for name in ('verify_storage', 'verify_collection_storage', 'verify_filtering_storage',
                 'verify_title_storage', 'verify_enrichment_evidence_storage'):
        checks.append((name, getattr(database, name)))
    schemas = importlib.import_module('schemas')
    checks.extend([('legacy schemas', schemas.verify_filtering_schemas),
                   ('title schemas', schemas.verify_title_schemas), ('reservations across drafts/runs', reservations)])
    if args.database:
        checks.append(('supplied database compatibility', lambda: compatibility(args.database, args.run_id)))
    failed = 0
    with patch.object(socket.socket, 'connect', blocked), patch.object(socket, 'create_connection', blocked):
        for name, check in checks:
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = check()
                print('PASS:', name, json.dumps(result) if result else '')
            except Exception as error:
                failed += 1
                # Do not expose saved contacts, draft text or secret values on failure.
                print('FAIL:', name, type(error).__name__)
    print(f'{len(checks) - failed}/{len(checks)} groups passed. Offline only; Gmail mocked.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

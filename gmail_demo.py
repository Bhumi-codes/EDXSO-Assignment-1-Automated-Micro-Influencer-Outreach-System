"""Optional Gmail test to an explicit user recipient; independent of simulation.

Sending is disabled until the commented main invocation at the bottom is enabled.
Use --authorize first, then --preview, then --send-test with an approved message ID.
Never use this demo to send directly to the saved influencer contact.
"""
from config import database_path, load_environment
import argparse
import base64
from contextlib import closing
from email.message import EmailMessage
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from review import connect, get_approved_emails
from sending import timestamp


SCOPES=['https://www.googleapis.com/auth/gmail.send']


def gmail_service(directory):
    # Lazy imports: simulation and previews need no Google dependencies.
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    directory=Path(directory); token=directory/'token.json'
    credentials=Credentials.from_authorized_user_file(str(token),SCOPES) if token.exists() else None
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow=InstalledAppFlow.from_client_secrets_file(str(directory/'credentials.json'),SCOPES)
            credentials=flow.run_local_server(port=0)
        token.write_text(credentials.to_json(),encoding='utf-8')
    if not credentials.has_scopes(SCOPES):
        raise ValueError('OAuth token lacks gmail.send; authorize again with the required scope.')
    return build('gmail','v1',credentials=credentials,cache_discovery=False)


def recipient_address(value):
    # Test address is new user input, not a revalidation of the creator email.
    from email.headerregistry import Address
    if any(character.isspace() for character in value):
        raise ValueError('Test recipient must be one address without whitespace.')
    parsed=Address(addr_spec=value)
    if not parsed.username or not parsed.domain:
        raise ValueError('Enter a complete test email address.')
    return value


def approved(path,run_id,message_id):
    records=[row for row in get_approved_emails(path,run_id) if row['message_id']==message_id]
    if len(records)!=1: raise ValueError('Choose a current approved email with a saved FOUND contact.')
    return records[0]


def send_test(path,run_id,message_id,recipient,service):
    recipient=recipient_address(recipient)
    with closing(connect(path,True)) as connection, connection:
        connection.execute('BEGIN IMMEDIATE')
        draft=approved(path,run_id,message_id)
        campaign=connection.execute('SELECT campaign_id FROM personalized_messages WHERE message_id=?',(message_id,)).fetchone()[0]
        connection.execute('''CREATE TABLE IF NOT EXISTS gmail_test_log(
            id INTEGER PRIMARY KEY, channel_id TEXT, campaign_id TEXT, test_recipient TEXT,
            status TEXT NOT NULL, record_json TEXT NOT NULL,
            UNIQUE(channel_id,campaign_id,test_recipient))''')
        key=(draft['channel_id'],campaign,recipient.casefold())
        if connection.execute('SELECT 1 FROM gmail_test_log WHERE channel_id=? AND campaign_id=? AND test_recipient=?',key).fetchone():
            return {'status':'SKIPPED_DUPLICATE_OR_UNCERTAIN','sent':False,'detail':'Existing Gmail test attempt; inspect its log before any manual recovery.'}
        row=dict(draft,mode='GMAIL_TEST',actual_recipient=recipient,intended_creator_recipient=draft['recipient'],date=timestamp(),sent=False,status='IN_PROGRESS',gmail_message_id=None)
        connection.execute('INSERT INTO gmail_test_log(channel_id,campaign_id,test_recipient,status,record_json) VALUES(?,?,?,?,?)',(*key,row['status'],json.dumps(row)))
        # Commit a durable reservation before a non-idempotent network request.
    message=EmailMessage(); message['To']=recipient; message['Subject']=draft['subject']; message.set_content(draft['body'])
    try:
        result=service.users().messages().send(userId='me',body={'raw':base64.urlsafe_b64encode(message.as_bytes()).decode('ascii')}).execute(num_retries=0)
        if not isinstance(result,dict) or not result.get('id'): raise ValueError('No Gmail message ID returned.')
        row.update(status='SENT',sent=True,gmail_message_id=result['id'],sent_at=timestamp(),detail='Gmail accepted the test message; inbox placement is not verified.')
    except Exception as error:
        # A timeout can happen after Gmail accepted a send. Never retry automatically.
        row.update(status='UNKNOWN',sent=False,detail=f'Outcome uncertain ({type(error).__name__}); inspect Gmail Sent before retrying.')
    with closing(connect(path,True)) as connection, connection:
        connection.execute('UPDATE gmail_test_log SET status=?,record_json=? WHERE channel_id=? AND campaign_id=? AND test_recipient=?',(row['status'],json.dumps(row),*key))
    return row


def test_log(path):
    with closing(connect(path)) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='gmail_test_log'").fetchone(): return []
        return [json.loads(row[0]) for row in connection.execute('SELECT record_json FROM gmail_test_log ORDER BY id')]


def self_check():
    draft=dict(channel_id='c',message_id='m',recipient='creator@example.test',subject='Saved subject',body='Saved body',review_id=1)
    with TemporaryDirectory() as directory:
        path=Path(directory)/'test.db'
        with closing(sqlite3.connect(path)) as connection,connection:
            connection.execute('CREATE TABLE personalized_messages(message_id TEXT,campaign_id TEXT)')
            connection.execute("INSERT INTO personalized_messages VALUES('m','test')")
        service=MagicMock()
        service.users.return_value.messages.return_value.send.return_value.execute.return_value={'id':'synthetic-gmail-id'}
        with patch(__name__+'.approved',return_value=draft):
            result=send_test(path,'test','m','tester@example.test',service)
            assert result['status']=='SENT' and result['actual_recipient']=='tester@example.test'
            assert result['intended_creator_recipient']=='creator@example.test'
            assert send_test(path,'test','m','tester@example.test',service)['status']=='SKIPPED_DUPLICATE_OR_UNCERTAIN'
            assert service.users.return_value.messages.return_value.send.call_count==1
            service.users.return_value.messages.return_value.send.return_value.execute.side_effect=TimeoutError()
            assert send_test(path,'test','m','other@example.test',service)['status']=='UNKNOWN'
            assert send_test(path,'test','m','other@example.test',service)['status']=='SKIPPED_DUPLICATE_OR_UNCERTAIN'
            assert len(test_log(path))==2
    print('Gmail checks OK: explicit test recipient, separate log, success, duplicate reservation and uncertain outcome; mocked service only.')


def main():
    load_environment()
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--authorize',action='store_true'); modes.add_argument('--preview',action='store_true'); modes.add_argument('--send-test',action='store_true'); modes.add_argument('--tracker',action='store_true')
    parser.add_argument('--run-id'); parser.add_argument('--message-id'); parser.add_argument('--test-recipient')
    directory=Path(__file__).resolve().parent
    parser.add_argument('--credentials-dir',default=str(directory))
    parser.add_argument('--database',default=str(database_path()))
    args=parser.parse_args()
    try:
        if args.authorize:
            gmail_service(args.credentials_dir); print('Gmail authorization completed. No email sent.'); return 0
        if args.tracker:
            print(json.dumps(test_log(args.database),indent=2,ensure_ascii=False)); return 0
        if not all((args.run_id,args.message_id,args.test_recipient)): parser.error('--run-id, --message-id and --test-recipient are required')
        recipient=recipient_address(args.test_recipient)
        draft=approved(args.database,args.run_id,args.message_id)
        if args.preview:
            print(f"TEST ONLY: To {recipient}\nSubject: {draft['subject']}\n{draft['body']}\nNo email sent; no OAuth request."); return 0
        result=send_test(args.database,args.run_id,args.message_id,recipient,gmail_service(args.credentials_dir))
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return 0 if result['status'] in ('SENT','SKIPPED_DUPLICATE_OR_UNCERTAIN') else 2
    except (OSError,ValueError,KeyError,sqlite3.Error,ImportError) as error:
        print(f'Gmail demo failed: {type(error).__name__}. Check local setup and arguments.'); return 1


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--self-check']:
        self_check()
        raise SystemExit(0)
    # Uncomment the following invocation ONLY when explicitly testing Gmail.
    # raise SystemExit(main())
    print('Gmail demo disabled. Uncomment the main invocation to enable explicit CLI commands. No email sent.')

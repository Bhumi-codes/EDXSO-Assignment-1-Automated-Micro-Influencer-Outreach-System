"""Simulate approved outreach and track it; never calls Gmail or Instagram."""
from config import database_path, load_environment
import argparse
from contextlib import closing
import csv
from datetime import datetime, timezone
import json
import sqlite3
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path
from review import connect, decision, fingerprint, load_profiles


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def run(path, run_id, limit=50, simulate=False):
    with closing(connect(path, simulate)) as connection, connection:
        if simulate:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('''CREATE TABLE IF NOT EXISTS simulation_outreach_log(
                id INTEGER PRIMARY KEY, channel_id TEXT, campaign_id TEXT, medium TEXT,
                run_id TEXT, record_json TEXT NOT NULL,
                UNIQUE(channel_id,campaign_id,medium))''')
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='simulation_outreach_log'").fetchone()
        outcomes = []
        for profile, draft in load_profiles(connection, run_id, limit):
            for medium in ('email','dm'):
                approval = decision(connection,profile,draft,medium)
                recipient = profile.get('contact_email') if medium == 'email' and profile['email_status']=='FOUND' else None
                if medium == 'dm':
                    recipient = next(iter(profile.get('instagram_urls',[])),None)
                key = (draft['channel_id'],draft['campaign_id'],medium)
                prior = connection.execute('SELECT record_json FROM simulation_outreach_log WHERE channel_id=? AND campaign_id=? AND medium=?',key).fetchone() if exists else None
                status = 'BLOCKED_NO_RECIPIENT' if not recipient else 'READY' if approval and approval['verdict']=='APPROVED' else 'BLOCKED_REVIEW'
                row = dict(influencer=profile['influencer_name'],recipient=recipient,channel_id=key[0],campaign_id=key[1],medium=medium,run_id=run_id,message_generated=True,sent=False,date=timestamp(),status=status,message_id=draft['message_id'],review_id=approval['review_id'] if approval else None,review_fingerprint=fingerprint(profile,draft,medium),subject=draft['response']['email_subject'] if medium=='email' else None,body=draft['response']['email_body'] if medium=='email' else draft['response']['instagram_dm'])
                if prior and json.loads(prior[0])['status']=='SIMULATED':
                    row['status']='SKIPPED_DUPLICATE'
                elif simulate:
                    if status=='READY': row['status']='SIMULATED'
                    connection.execute('INSERT INTO simulation_outreach_log(channel_id,campaign_id,medium,run_id,record_json) VALUES(?,?,?,?,?) ON CONFLICT(channel_id,campaign_id,medium) DO UPDATE SET run_id=excluded.run_id,record_json=excluded.record_json',(*key,run_id,json.dumps(row,ensure_ascii=False)))
                outcomes.append(row)
        return outcomes


def tracker(path,run_id):
    with closing(connect(path)) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='simulation_outreach_log'").fetchone(): return []
        return [json.loads(row[0]) for row in connection.execute('SELECT record_json FROM simulation_outreach_log WHERE run_id=? ORDER BY id',(run_id,))]


def export(rows,path):
    fields=['influencer','recipient','medium','message_generated','sent','date','status','message_id','review_id','subject','body']
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
        for row in rows:
            row=dict(row); row['sent']='Yes' if row['sent'] else 'No'; row['message_generated']='Yes' if row['message_generated'] else 'No'
            for key,value in row.items():
                if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')): row[key]="'"+value
            writer.writerow(row)


def self_check():
    profile=dict(influencer_name='Synthetic',email_status='FOUND',contact_email='observed@example.test',instagram_urls=[])
    draft=dict(channel_id='c',campaign_id='test',message_id='m',response=dict(email_subject='Saved subject',email_body='Saved body',instagram_dm='Saved DM'))
    with TemporaryDirectory() as directory:
        path=Path(directory)/'test.db'
        with closing(sqlite3.connect(path)) as connection:
            connection.execute('CREATE TABLE marker(id INTEGER)')
        with patch(__name__+'.load_profiles',return_value=[(profile,draft)]), patch(__name__+'.decision',return_value={'verdict':'APPROVED','review_id':1}), patch(__name__+'.fingerprint',return_value='exact'):
            assert run(path,'test',simulate=False)[0]['status']=='READY'
            assert tracker(path,'test')==[]
            rows=run(path,'test',simulate=True)
            assert [row['status'] for row in rows]==['SIMULATED','BLOCKED_NO_RECIPIENT']
            assert rows[0]['body']=='Saved body' and not rows[0]['sent']
            assert run(path,'test',simulate=True)[0]['status']=='SKIPPED_DUPLICATE'
            profile['instagram_urls']=['https://instagram.com/synthetic']
            assert run(path,'test',simulate=True)[1]['status']=='SIMULATED'
            assert len(tracker(path,'test'))==2
            export(tracker(path,'test'),Path(directory)/'tracker.csv')
    print('Sending checks OK: read-only preview, exact messages, email/DM simulation, missing recipients, duplicate prevention and tracker export.')
    print('Synthetic data, mocked review inputs and temporary database only; no real messages or requests.')


def main():
    load_environment()
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--preview',action='store_true'); modes.add_argument('--simulate',action='store_true'); modes.add_argument('--tracker',action='store_true')
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--run-id'); parser.add_argument('--limit',type=int,default=50); parser.add_argument('--export')
    parser.add_argument('--database',default=str(database_path()))
    args=parser.parse_args()
    if args.self_check: self_check(); return 0
    if not args.run_id: parser.error('--run-id is required')
    if args.limit<1: parser.error('--limit must be positive')
    if args.export and not args.tracker: parser.error('--export requires --tracker')
    try:
        rows=tracker(args.database,args.run_id) if args.tracker else run(args.database,args.run_id,args.limit,args.simulate)
        for row in rows:
            print(f"{row['influencer']} | {row['medium']} | {row['recipient'] or 'No recipient'} | {row['status']} | Sent: No")
            if row['status'] in ('READY','SIMULATED'): print(f"{row['subject'] or 'DM'}\n{row['body']}\n")
        if args.export: export(rows,args.export)
        print('Simulation only. No network requests or real messages sent.')
        return 0
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as error:
        print(f'Simulation failed: {error}'); return 1


if __name__=='__main__':
    raise SystemExit(main())

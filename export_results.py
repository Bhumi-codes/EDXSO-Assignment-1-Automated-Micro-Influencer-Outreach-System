"""Export saved cohort results. Read-only database access; no network or sending.

Usage: python export_results.py --run-id RUN_ID --output-dir exports
Missing enrichment is NOT_ENRICHED, not an unsuccessful contact search.
The README is intentionally handled separately and is never modified here.
"""
from config import database_path, load_environment
import argparse
from collections import Counter
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from review import connect, decision


def has_table(connection, name):
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(name,)).fetchone() is not None


def decoded(row):
    return json.loads(row['record_json']) if row else None


def latest(connection, table, clause, parameters, order):
    if not has_table(connection,table): return None
    return decoded(connection.execute(f'SELECT record_json FROM {table} WHERE {clause} ORDER BY {order} DESC LIMIT 1',parameters).fetchone())


def collect(connection, run_id):
    collections=connection.execute('SELECT channel_id,record_json FROM collection_results WHERE run_id=? ORDER BY channel_id',(run_id,)).fetchall()
    if not collections: raise ValueError('No saved collection cohort found for this run ID.')
    dataset=[]; profiles=[]; messages=[]
    for sample_row in collections:
        channel_id=sample_row['channel_id']
        sample=decoded(sample_row)
        channel=decoded(connection.execute('SELECT record_json FROM influencers WHERE channel_id=?',(channel_id,)).fetchone())
        if channel is None or channel.get('channel_id')!=channel_id:
            raise ValueError('Collection channel is missing or inconsistent.')
        filtering=latest(connection,'filtering_results','collection_run_id=? AND channel_id=?',(run_id,channel_id),'evaluated_at DESC, filtering_id')
        profile=None; classification=None
        if filtering:
            if filtering['channel_id']!=channel_id or filtering['collection_run_id']!=run_id:
                raise ValueError('Filtering links do not match the requested cohort.')
            classification_id=filtering.get('classification_id')
            if classification_id:
                classification=latest(connection,'classifications','classification_id=?',(classification_id,),'classified_at')
            if filtering['status']=='PASS':
                profile=latest(connection,'profile_enrichments','filtering_id=?',(filtering['filtering_id'],),'enriched_at DESC, enrichment_id')
        if profile:
            if any(profile.get(key)!=value for key,value in (('channel_id',channel_id),('collection_run_id',run_id),('filtering_id',filtering['filtering_id']))):
                raise ValueError('Enrichment links do not match the current PASS result.')
            profiles.append(profile)
            draft=latest(connection,'personalized_messages','enrichment_id=?',(profile['enrichment_id'],),'generated_at DESC, message_id')
            if draft:
                if any(draft.get(key)!=value for key,value in (('channel_id',channel_id),('collection_run_id',run_id),('filtering_id',filtering['filtering_id']))):
                    raise ValueError('Draft links do not match its profile.')
                reviews={}
                for medium in ('email','dm'):
                    reviewed=decision(connection,profile,draft,medium)
                    reviews[medium]=dict(reviewed) if reviewed else {'verdict':'PENDING_REVIEW'}
                messages.append(dict(name=profile['influencer_name'],profile=profile,draft=draft,reviews=reviews))
        engagement=(filtering or {}).get('engagement',{})
        technology=(filtering or {}).get('technology',{})
        niche=profile.get('category') if profile else technology.get('label')
        if not niche and classification:
            niche=classification.get('response',{}).get('niche',{}).get('label')
        reasons=list((filtering or {}).get('reasons',[]))
        for criterion in ('technology','content_relevance','subscribers','engagement'):
            for reason in (filtering or {}).get(criterion,{}).get('reasons',[]):
                entry=f'{criterion}: {reason}'
                if entry not in reasons: reasons.append(entry)
        email_status=profile['email_status'] if profile else 'NOT_ENRICHED'
        email=(profile.get('contact_email') if email_status=='FOUND' else profile.get('email_display') or email_status) if profile else 'Not enriched'
        dataset.append(dict(Name=channel['title'],Platform='YouTube',Followers=channel.get('subscriber_count') if channel.get('subscriber_count') is not None else 'Unavailable',Engagement_Percent=engagement.get('rate_percent') if engagement.get('rate_percent') is not None else 'Unavailable',Niche=niche or 'Unavailable',Email=email,Profile_URL=channel['profile_url'],Content_Theme='; '.join(theme['name'] for theme in profile['content_themes']) if profile else 'Not enriched',Status=filtering['status'] if filtering else 'NOT_EVALUATED',Reasons='; '.join(reasons),Email_Status=email_status,Channel_ID=channel_id,Collection_Run_ID=run_id,Collection_Status=sample['status'],Sample_Video_Count=len(sample.get('selected_video_ids',[])),Channel_Metadata_Observed_At=channel.get('collected_at'),Filtering_Evaluated_At=(filtering or {}).get('evaluated_at'),Engagement_Formula=engagement.get('formula_version'),Data_Use_Permission=engagement.get('data_use_permission'),Data_Use_Note=engagement.get('data_use_note')))
    simulation=[]; gmail=[]
    if has_table(connection,'simulation_outreach_log'):
        simulation=[decoded(row) for row in connection.execute('SELECT record_json FROM simulation_outreach_log WHERE run_id=? ORDER BY id',(run_id,))]
    if has_table(connection,'gmail_test_log'):
        gmail=[decoded(row) for row in connection.execute('SELECT record_json FROM gmail_test_log ORDER BY id')]
        gmail=[row for row in gmail if row.get('collection_run_id')==run_id]
    return dataset,profiles,messages,simulation,gmail


def safe_cell(value):
    if value is None: return ''
    value=str(value)
    return "'"+value if value.lstrip().startswith(('=','+','-','@')) else value


def write_csv(path,fields,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows: writer.writerow({field:safe_cell(row.get(field)) for field in fields})


def tracking_rows(rows, mode):
    result=[]
    for row in rows:
        sent='Yes' if row.get('sent') else 'Unknown' if row['status'] in ('UNKNOWN','IN_PROGRESS') else 'No'
        result.append(dict(Influencer=row.get('influencer') or row.get('channel_id'),Email=row.get('actual_recipient') if mode=='GMAIL_TEST' else row.get('recipient'),Message_Generated='Yes' if row.get('message_id') else 'No',Sent=sent,Date=row.get('sent_at') or row.get('date'),Status=row['status'],Mode=mode,Channel=row.get('medium','email'),Intended_Creator_Recipient=row.get('intended_creator_recipient'),Channel_ID=row.get('channel_id'),Campaign_ID=row.get('campaign_id'),Message_ID=row.get('message_id'),Review_ID=row.get('review_id'),Gmail_Message_ID=row.get('gmail_message_id'),Subject=row.get('subject'),Body=row.get('body'),Detail=row.get('detail')))
    return result


def export_results(path,run_id,output_dir):
    with closing(connect(path)) as connection:
        connection.execute('BEGIN')  # All exports use one consistent read snapshot.
        data,profiles,messages,simulation,gmail=collect(connection,run_id)
        for row in gmail:
            name=next((item['Name'] for item in data if item['Channel_ID']==row['channel_id']),row['channel_id'])
            row['influencer']=name
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'influencers.csv',list(data[0]),data)
    (output/'shortlisted_profiles.json').write_text(json.dumps(profiles,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    text='# Saved personalized outreach messages\n\nRun: '+run_id+'\n\nThese are saved drafts. Gmail test recipients are reported separately in the Gmail tracker.\n'
    for item in messages:
        draft=item['draft']; response=draft['response']; profile=item['profile']
        text+=f"\n## {item['name']}\n\nMessage ID: {draft['message_id']}\n\nEmail review: {item['reviews']['email']['verdict']}; DM review: {item['reviews']['dm']['verdict']}\n\nSaved email: {profile.get('contact_email') or profile['email_status']}\n\nEmail subject: {response['email_subject']}\n\nEmail ({draft['email_word_count']} words):\n\n"
        text+='\n'.join('    '+line for line in response['email_body'].split('\n'))+'\n'
        text+=f"\nInstagram DM ({draft['dm_word_count']} words):\n\n"
        text+='\n'.join('    '+line for line in response['instagram_dm'].split('\n'))+'\n'
        text+=f"\nProvider/model: {draft.get('provider','historical')} / {draft['model']}; prompt: {draft['prompt_version']}\n"
    (output/'personalized_messages.md').write_text(text,encoding='utf-8')
    fields=['Influencer','Email','Message_Generated','Sent','Date','Status','Mode','Channel','Intended_Creator_Recipient','Channel_ID','Campaign_ID','Message_ID','Review_ID','Gmail_Message_ID','Subject','Body','Detail']
    write_csv(output/'simulation_tracker.csv',fields,tracking_rows(simulation,'SIMULATION'))
    write_csv(output/'gmail_test_tracker.csv',fields,tracking_rows(gmail,'GMAIL_TEST'))
    summary=dict(collection_run_id=run_id,exported_at=datetime.now(timezone.utc).isoformat(),cohort_influencers=len(data),minimum_50_met=len(data)>=50,filtering_status_counts=dict(Counter(row['Status'] for row in data)),enriched_current_PASS_profiles=len(profiles),selected_emails_found=sum(p['email_status']=='FOUND' for p in profiles),personalized_draft_pairs=len(messages),simulation_status_counts=dict(Counter(row['status'] for row in simulation)),gmail_test_status_counts=dict(Counter(row['status'] for row in gmail)),database_writes=False,network_requests=0,notes=['Channel metadata is current saved data, not a per-run snapshot.','Only enrichment linked to the current PASS filtering result is exported.','Not enriched does not mean an email search returned Not Found.','Simulation is not a real send; Gmail SENT means API acceptance, not automated inbox verification.','CSV formula escaping changes presentation only; JSON and database retain original values.'])
    summary['files_sha256']={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('influencers.csv','shortlisted_profiles.json','personalized_messages.md','simulation_tracker.csv','gmail_test_tracker.csv')}
    (output/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


def self_check():
    with TemporaryDirectory() as directory:
        path=Path(directory)/'test.db'
        with closing(sqlite3.connect(path)) as connection,connection:
            connection.executescript('CREATE TABLE influencers(channel_id TEXT,record_json TEXT); CREATE TABLE collection_results(run_id TEXT,channel_id TEXT,record_json TEXT);')
            for number in range(50):
                channel=dict(channel_id=str(number),title='Synthetic '+str(number),subscriber_count=0 if number==0 else None,profile_url='https://example.test/'+str(number))
                connection.execute('INSERT INTO influencers VALUES(?,?)',(str(number),json.dumps(channel)))
                connection.execute('INSERT INTO collection_results VALUES(?,?,?)',('check',str(number),json.dumps(dict(status='COMPLETE',selected_video_ids=[]))))
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        summary=export_results(path,'check',Path(directory)/'exports')
        assert summary['cohort_influencers']==50 and summary['minimum_50_met'] and summary['personalized_draft_pairs']==0
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        with (Path(directory)/'exports'/'influencers.csv').open(encoding='utf-8-sig',newline='') as handle:
            rows=list(csv.DictReader(handle))
        assert rows[0]['Followers']=='0' and rows[1]['Followers']=='Unavailable' and rows[0]['Email_Status']=='NOT_ENRICHED'
        assert safe_cell('=formula')=="'=formula"
        assert tracking_rows([dict(channel_id='c',status='UNKNOWN',sent=False)],'GMAIL_TEST')[0]['Sent']=='Unknown'
    print('Export checks OK: 50-row cohort, missing/zero values, optional stages, CSV escaping and read-only database.')
    print('Synthetic temporary database only; no network, LLM, sending or project database changes.')


def main():
    load_environment()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true'); parser.add_argument('--run-id')
    parser.add_argument('--database',default=str(database_path()))
    parser.add_argument('--output-dir',default=str(Path(__file__).resolve().parent/'exports'))
    args=parser.parse_args()
    if args.self_check: self_check(); return 0
    if not args.run_id: parser.error('--run-id is required')
    try:
        summary=export_results(args.database,args.run_id,args.output_dir)
        print(json.dumps({key:value for key,value in summary.items() if key not in ('files_sha256','notes')},indent=2))
        print(f'Exports saved to {Path(args.output_dir).resolve()}. README unchanged. No requests, database writes or messages sent.')
        return 0
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as error:
        print(f'Export failed: {error}'); return 1


if __name__=='__main__':
    raise SystemExit(main())

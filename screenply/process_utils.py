from uuid import uuid4
import pandas as pd
import os
import json
import ocrmypdf
import datetime

from screenply.parser import Screenplay


def serialize(data):
    # replace NaN with null for valid JSON
    data = data.where((pd.notnull(data)), None)    
    return "\n".join([json.dumps(l) for l in data.to_dict(orient='records')])


def process(file, failure_path=None, ocr_output_dir=None, gcs_bucket=None, **kwargs):
    if file.endswith('.pdf'):
        if not ocr_output_dir:
            ocr_output_dir = 'ocr_output'
        if not os.path.exists(ocr_output_dir):
            os.mkdir(ocr_output_dir)
        ocr_output_path = os.path.join(ocr_output_dir, os.path.basename(file))
        ocrmypdf.ocr(
            input_file=file, 
            output_file=ocr_output_path, 
            deskew=True, 
            use_threads=True,
            skip_text=True,
        )
        source = ocr_output_path
    else:
        source = file
    scr = Screenplay(
        source=source, 
        failure_path=failure_path,
        **kwargs
    )
    if gcs_bucket:
        gcs_filename = "{}.json".format(scr.title)
        gcs_upload(scr.data, gcs_bucket, gcs_filename)
    return scr


def failure(file, e, failure_path):
    now = str(datetime.datetime.now())
    failure_path = failure_path or "%s_failures.json" % uuid4()
    failure_info = {'title': file, 'date': now, 'reason': str(e), 'value': 'process'}
    with open(failure_path, 'a') as f:
        f.write(json.dumps(failure_info) + "\n")


def try_process(file, failure_path=None, ocr_output_dir=None, gcs_bucket=None, **kwargs):     
    try:
        return process(
            file, 
            failure_path=failure_path,
            ocr_output_dir=ocr_output_dir,
            **kwargs
        )
    except Exception as e:
        failure(file, e, failure_path)


def gcs_upload(data, gcs_bucket, gcs_filename):
    if len(data) > 0:
        blob = gcs_bucket.blob(gcs_filename)
        blob.upload_from_string(serialize(data))
    msg = """
    wrote {gcs_filename}
    in bucket: {bucket}
    """
    print(msg.format(gcs_filename=gcs_filename, bucket=gcs_bucket))


def scan_folder(folder, suffix='.pdf'):
    result = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if suffix in name:
                result.append(path)
    return result


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

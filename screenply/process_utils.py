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


def process(file, failure_path=None, ocr_output_dir=None, **kwargs):
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
    return scr


def batch_process(files, failure_path=None, ocr_output_dir=None, **kwargs):
    if not failure_path:
        failure_path = "%s_failures.json" % uuid4()
    data = pd.DataFrame()
    for file in files:
        try:
            scr = process(
                file, 
                failure_path=failure_path,
                ocr_output_dir=ocr_output_dir,
                **kwargs
            )
            data = data.append(scr.data)
        except FileNotFoundError as e:
            now = str(datetime.datetime.now())
            failure_info = {'title': file, 'date': now, 'reason': str(e), 'value': 'process'}
            if failure_path:
                with open(failure_path, 'a') as f:
                    f.write(json.dumps(failure_info) + "\n")
    return data


def batch_upload(files, bucket, gcs_filename, failure_path=None, html=False, debug_mode=False):
    data = batch_process(files, failure_path=failure_path, html=html)
    if len(data) > 0:
        blob = bucket.blob(gcs_filename)
        blob.upload_from_string(serialize(data))
    msg = """
    wrote {n_files} to {gcs_filename}
    in bucket: {bucket}
    """
    print(msg.format(n_files=data.title.nunique, gcs_filename=gcs_filename, bucket=bucket))


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

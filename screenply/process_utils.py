import pandas as pd
import os
import json

from parser import Screenplay


def serialize(data):
    # replace NaN with null for valid JSON
    data = data.where((pd.notnull(data)), None)    
    return "\n".join([json.dumps(l) for l in data.to_dict(orient='records')])


def batch_process(files, failure_path=None, html=False, debug_mode=False):
	if not failure_path:
		failure_path = "%s_failures.json" % uuid4()
	data = pd.DataFrame()
	for f in files:
		try:
			if html:
				scr = Screenplay(from_html=f, debug_mode=debug_mode)
			else:
				scr = Screenplay(f, debug_mode=debug_mode)
			data = data.append(scr.data)
			if failure_path and scr.failure_info:
				with open(failure_path, 'a') as f:
					f.write(json.dumps(scr.failure_info)+"\n")
		except:
			print("%s failed" % f)
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

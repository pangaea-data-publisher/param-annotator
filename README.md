# param-annotator
PANGAEA semantic annotator for parameters

## Usage
Minimum Python 3.7+

This service requires running the PUCUM web service running on same machine, it must be started before!
See https://github.com/pangaea-data-publisher/pucum on how to download and setup.

In addition, stopwords from nltk should be downloaded prior to running the service:
```
python3 -m nltk.downloader stopwords
```

To run the service (default port 8383), please execute the following from the root directory:

```
pip3 install -r requirements.txt
python3 api/appv1.py
```

To get semantic terms associated with a parameter ('unit' is optional).
```
http://localhost:8383/param-annotator/api?name={parameter_name}&unit={units_of_measurements}
```

# param-annotator
PANGAEA semantic annotator for parameters

## Usage
Python 3.7+

To run the service (default port 8383), please execute the following from the root directory:

```
pip install -r requirements.txt
python3 api/appv1.py
```

To get semantic terms associated with a parameter ('unit' is optional).
```
http://localhost:8383/param-annotator/api?name={parameter_name}&unit={units_of_measurements}
```

Stopwords from nltk should be download prior to running the service
```
python3 -m nltk.downloader stopwords
```

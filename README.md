# param-annotator
PANGAEA semantic annotator for parameters

## Usage
Python 3.7+

To run the service (default port 8383), please execute the following from the root directory:

```
pip install -r requirements.txt
python api/appv1.py
```

Stopwords from nltk should be download prior to running the service
```
import nltk
nltk.download('stopwords')
```


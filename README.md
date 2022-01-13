
##Code for synthetic data generation for the paper "Multi-head Sequence Tagging Model for Grammatical Error Correction"## 
## Installation

Requires Python 3.

```bash
# apt-get packages (required for hunspell & pattern)
apt-get update
apt-get install libhunspell-dev libmysqlclient-dev -y

# pip packages
pip install --upgrade pip
pip install -U word_forms
pip install --upgrade -r requirements.txt
python -m spacy download en

# pattern3 (see https://www.clips.uantwerpen.be/pages/pattern for any installation issues)
pip install pattern3
python -c "import site; print(site.getsitepackages())"
# ['PATH_TO_SITE_PACKAGES']
cp tree.py PATH_TO_SITE_PACKAGES/pattern3/text/
```

## Preprocess Data

```bash
python preprocess.py
```

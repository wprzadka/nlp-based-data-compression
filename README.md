# nlp-based-data-compression

### possible arguments:
```
-v --version
    Prints current program version
-h --help
    Prints arguments informations
-e --encode file_path
    Encodes file indicated by "file_path"
-d --decode file_path
    Decodes file indicated by "file_path"
```

### installation of necessary libraries
```bash
python3 -m venv venv
source ./venv/bin/activate
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install transformers
```
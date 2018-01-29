## Installation
```
sudo pip install virtualenv           # This may already be installed
virtualenv .env                       # Create a virtual environment
source .env/bin/activate              # Activate the virtual environment
pip install -r requirements.txt       # Install dependencies
```

## Data preparation
Two steps should be performed to prepare data:

+ run the `download_data.sh` to get the train data
+ execute the `tokenizer.sh` to clean up the data for training

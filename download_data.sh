# a HUGE thank to the person who wrote this: https://medium.com/geekculture/wget-large-files-from-google-drive-336ba2e1c991

# exits if a command fails
set -e

DIRECTORY='./data/'

if [ ! -e "$DIRECTORY" ]; then
  echo Creating a $DIRECTORY folder...
    mkdir data
fi

echo Fetching data...
FILENAME="./data/utterances.jsonl.zip"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RO631Tojr5yoDpsyCLH-eXdSGCf5TKBW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RO631Tojr5yoDpsyCLH-eXdSGCf5TKBW" -O $FILENAME && rm -rf /tmp/cookies.txt

echo Unzip data...
unzip $FILENAME -d ./data/ && rm "$FILENAME"
# Exits if a command fails
set -e

# Downloading data
echo Fetching data...
FILENAME="/data/urlsf_subset00.tar"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/u/0/uc?export=download&id=1uQEO2XayD2c2QJ0khpNYLNA6iKJ9SZ60' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RO631Tojr5yoDpsyCLH-eXdSGCf5TKBW" -O $FILENAME && rm -rf /tmp/cookies.txt

# Unzip the .tar archive and go inside the data folder
tar -xf $FILENAME
cd data

# Process the first 50 archive files
ls urlsf_subset00-{0..50}_data.xz |xargs -n1 tar -xf

# Display how many files in the folder
NFILES= ls | wc -l
echo "$NFILES has been processed"

# Convert all data files in txt files
for f in *; do case "$f" in *.txt) echo skipped $f;; *) mv "$f" "$f".txt; esac; done
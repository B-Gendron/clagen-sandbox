# Exits if a command fails
set -e

# Downloading data
echo Fetching data...
FILENAME="urlsf_subset00.tar"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/u/0/uc?export=download&id=1uQEO2XayD2c2QJ0khpNYLNA6iKJ9SZ60' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uQEO2XayD2c2QJ0khpNYLNA6iKJ9SZ60" -O $FILENAME && rm -rf /tmp/cookies.txt

# Unzip the .tar archive and go inside the data folder
echo Unzip .tar archive...
tar -xf $FILENAME
cd openwebtext

# Process the first 50 archive files
echo Process the first 50 archive files...
ls urlsf_subset00-{0..50}_data.xz |xargs -n1 tar -xf

# Convert all data files in txt files
echo Convert archive files into .txt files...
for f in *; do case "$f" in *.txt) echo skipped $f;; *) mv "$f" "$f".txt; esac; done

# Delete all .xz achives as we only want to keep the .txt files from the 50 first archives in this case (in order to alleviate training data and account for a reasonable amount of computation time)
echo Delete unused files...
rm *.xz.txt
rm ../$FILENAME

# Merge all txt files into a big one
cat *.txt >> data.txt
rm 0*.txt 

# Move data to the right location
cd ..
mv ./openwebtext ..
#!/bin/bash
echo "Installing all necessary dependencies..."
    pip install -r requirements.txt
echo "Done with installs!"
echo "Pulling BRATS 2018 Data down, this may take a few minutes..."
if command -v python3 &>/dev/null; then
  python3 Utils/getBraTs2018Data.py
else
  python Utils/getBraTS2018Data.py
fi
echo "Success! You should be good to go! :)" 



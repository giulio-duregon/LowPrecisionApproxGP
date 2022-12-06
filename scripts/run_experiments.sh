# For data set bikes, run with base kernel
python model_runner.py -d bikes -l true -dt double -ip 50 -it 50 -dt double
python model_runner.py -d bikes -l true -dt double -ip 50 -it 50 -dt single

# Parse the output logs
echo "Parsing logs!"
python scripts/parse_logs.py

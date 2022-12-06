# For data set bikes, run with base kernel
python model_runner.py -d bikes -l true -dt double -ip 50 -it 50 -dt double -m true -j 100
python model_runner.py -d bikes -l true -dt double -ip 50 -it 50 -dt single -m true -j 100
python model_runner.py -d bikes -l true -dt double -ip 100 -it 100 -dt double -m true -j 100
python model_runner.py -d bikes -l true -dt double -ip 100 -it 100 -dt single -m true -j 100
python model_runner.py -d bikes -l true -dt double -ip 200 -it 200 -dt double -m true -j 100
python model_runner.py -d bikes -l true -dt double -ip 200 -it 200 -dt single -m true -j 100
python model_runner.py -d bikes -l true -dt double -ip 500 -it 500 -dt double -m true -j 100
python model_runner.py -d bikes -l true -dt double -ip 500 -it 500 -dt single -m true -j 100

# python model_runner.py -d bikes -l true -dt double -ip 100 -it 100 -dt single

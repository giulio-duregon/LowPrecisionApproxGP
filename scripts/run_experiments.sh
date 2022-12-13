# For data set bikes, run with base kernel

for dataset in "energy" "bikes" "protein" "naval"; do
    echo $dataset
    python model_runner.py -d $dataset -l true -ip 50 -it 50 -dt single
    python model_runner.py -d $dataset -l true -ip 50 -it 50 -dt double
done

# for dataset in "bikes" "protein" "naval" "road3d"; do
#     for ip in 50 75 100 150 200; do
#         for precision in "single" "double"; do
#             echo $dataset, $ip, $precision
#             python model_runner.py -d $dataset -l true -ip $ip -it $ip -dt $precision
#         done
#     done
# done

# for dataset in "bikes" "protein" "naval" "road3d"; do
#     for ip in 50 75 100 150 200; do
#         for precision in "single" "double"; do
#             for j in 10 20 50 100; do
#                 echo $dataset, $ip, $precision, $j
#                 python model_runner.py -d $dataset -l true -ip $ip -it $ip -dt $precision -m true -j $j -mj 10
#             done
#         done
#     done
# done

# Parse the output logs
echo "Parsing logs!"
python scripts/parse_logs.py

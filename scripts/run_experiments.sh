# Run experiments on small datasets with first greedy
for dataset in "bikes" "protein" "naval" "energy"; do
    for ip in 50 75 100 150 200; do
        for precision in "single" "double"; do
            echo $dataset, $ip, $precision
            python model_runner.py -d $dataset -l true -ip $ip -it $ip -dt $precision
        done
    done
done

# Run experiments on small datasets with max greedy
for dataset in "bikes" "protein" "naval" "energy"; do
    for ip in 50 75 100 150 200; do
        for precision in "single" "double"; do
            for j in 10 20 40 60; do
                echo $dataset, $ip, $precision, $j
                python model_runner.py -d $dataset -l true -ip $ip -it $ip -dt $precision -m true -j $j -mj 10
            done
        done
    done
done

# road3d last for both max/first greedy since it takes forever
for ip in 50 75 100 150 200; do
    for precision in "single" "double"; do
        echo "road3d", $ip, $precision
        python model_runner.py -d road3d -l true -ip $ip -it $ip -dt $precision
    done
done

for ip in 50 75 100 150 200; do
    for precision in "single" "double"; do
        for j in 10 20 40 60; do
            echo "road3d" $ip, $precision, $j
            python model_runner.py -d road3d -l true -ip $ip -it $ip -dt $precision -m true -j $j -mj 10
        done
    done
done

# Parse the output logs
echo "Parsing logs!"
python scripts/parse_logs.py

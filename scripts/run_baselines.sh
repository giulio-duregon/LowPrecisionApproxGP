# Run experiments on small datasets with first greedy
for seed in 5 10 15 20 25; do
    for dataset in "bikes" "protein" "energy"; do
        for it in 100; do
            echo seed: $seed, dataset: $dataset, iterations: $it
            python baseline_runner.py -d $dataset -dt single -it $it --seed $seed
        done
    done
done

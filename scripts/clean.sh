while getopts d:g: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        g) gen=${OPTARG};;
    esac
done
find "./data/${dataset}/${gen}_gen" -name "*.pkl" -delete
find "./data/${dataset}/${gen}_gen" -name "*.log" -delete
find "./data/${dataset}/${gen}_gen" -name "record*" -delete
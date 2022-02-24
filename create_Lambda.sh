# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
train_epoch=150
datasets=("CAVE" "Harvard" "ICVL")
model_names=("Lambda")
block_num=9
concats=('False' 'True')
loss_modes=("mse" "mse_sam")
start_time=$(date "+%m%d")


while getopts b:e:d:c:m:bn:s: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) train_epoch=$OPTARG ;;
        d) datasets=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        bn) block_num=$OPTARG ;;
        s) start_time=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


for dataset in $datasets; do
    for concat in $concats; do
        for loss_mode in $loss_modes; do
            for model_name in $model_names; do
                echo $dataset $concat $loss_mode $model_name
                python reconst.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $model_name
                python refine.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $model_name
                python evaluate_lambda.py -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -m $model_name

            done
        done
    done
done

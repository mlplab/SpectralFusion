# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
train_epoch=150
datasets=("CAVE Harvard ICVL")
base_model_name="SSAttention"
block_nums=()
for i in {1..13}; do
    block_nums+=($i)
done
concats=('True' 'False')
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
    for block_num in $block_nums; do
        for concat in $concats; do
            for loss_mode in $loss_modes; do

            python train_attention.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -l $loss_mode
            python evaluate_attention.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -l $loss_mode

        done
    done
done

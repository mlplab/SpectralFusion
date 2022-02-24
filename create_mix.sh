# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
train_epoch=150
datasets=("CAVE" "Harvard" "ICVL")
model_names=("HyperMix")
for i in {1..13}; do
    block_nums+=($i)
done
concats=('False' 'True')
loss_modes=('mse' 'mse_sam')
chuncks=(2 3 4)
start_time=$(date "+%m%d")
# start_time='1115'
# start_time=1009
# start_time='0702'


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


end_status=1

for dataset in $datasets; do
    for concat in $concats; do
        for block_num in $block_nums; do
            for loss_mode in $loss_modes; do
                for base_model_name in $model_names; do
                    for chunck in $chuncks; do

                        name_block_num=$(printf %02d $block_num)
                        model_name=$base_model_name\_$name_block_num\_$loss_mode\_$concat\_$chunck
                        echo $model_name

                        python train_mix.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -ch $chunck
                        python evaluate_mix.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -ch $chunck

                    done
                done
            done
        done
    done
done

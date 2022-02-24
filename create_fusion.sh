# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
train_epoch=150
datasets=("CAVE" "Harvard" "ICVL")
base_model_name="SpectralFusion"
block_nums=()
for i in {1..13}; do
    block_nums+=($i)
done
concats=('True' 'False')
modes=("inputOnly" "outputOnly" "both")
conv_modes=("normal" "edsr" "ghost")
loss_modes=("fusion" "fusion" "mse")
start_time=$(date "+%m%d")
# start_time='0915'


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
            for mode in $modes; do
                for conv_mode in $conv_modes; do
                    for loss_mode in $loss_modes; do
                        python train_fusion.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -md $mode -l $loss_mode -cm $conv_mode
                        python evaluate_fusion.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -md $mode -l $loss_mode -cm $conv_mode
                    done
                done
            done
        done
    done
done

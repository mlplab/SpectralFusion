# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
# search_epoch=100
train_epoch=150
datasets=("CAVE")
base_model_name="SpectralFusion"
block_nums=(3)
concats=('False')
modes=("inputOnly")
conv_modes=("normal" "edsr" "ghost", "cp")
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


i=1
for dataset in $datasets; do
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/SpectralFusion
    for block_num in $block_nums; do
        for concat in $concats; do
            for mode in $modes; do
                for conv_mode in $conv_modes; do
                    python train_fusion.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -md $mode -l $loss_modes[$i] -cm $conv_mode
                    python evaluate_fusion.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -md $mode -l $loss_modes[$i] -cm $conv_mode

                    name_block_num=$(printf %02d $block_num)
                    model_name=$base_model_name\_$name_block_num\_${loss_modes[$i]}\_$mode\_$start_time\_$concat\_$conv_mode
                    mkdir -p ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
                    cp ../SCI_result/$dataset\_$start_time/$model_name/output.csv ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/$model_name\_output.csv
                    cp ../SCI_ckpt/$dataset\_$start_time/all_trained/$model_name.tar ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/
                    skicka upload ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/SpectralFusion/$model_name/
                done
            done
        done
    done
done

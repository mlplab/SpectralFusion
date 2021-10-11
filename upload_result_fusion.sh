# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
# search_epoch=100
train_epoch=150
datasets=("CAVE" "Harvard")
base_model_name="SpectralFusion_EDSR"
block_nums=(1 2 3 4 5 6 7 8 9 10 11 12 13)
concats=('False' 'True')
modes=("inputOnly")
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
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/SpectralFusion
    for block_num in $block_nums; do
        for concat in $concats; do
            i=1
            for mode in $modes; do

                name_block_num=$(printf %02d $block_num)
                model_name=$base_model_name\_$name_block_num\_${loss_modes[$i]}\_$mode\_$start_time\_$concat
                mkdir ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
                cp ../SCI_result/$dataset\_$start_time/$model_name/output.csv ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/$model_name\_output.csv
                cp ../SCI_ckpt/$dataset\_$start_time/all_trained/$model_name.tar ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/
                skicka upload ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/SpectralFusion/$model_name/
                let i++
            done
        done
    done
done

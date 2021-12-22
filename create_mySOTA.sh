# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
# search_epoch=100
train_epoch=150
datasets=("ICVL")
model_names=("Attention" "HyperMix")
# model_names=("Attention" "HyperMix")
block_nums=()
for i in {5..9..2}; do
    block_nums+=($i)
done
concats=('False' 'True')
loss_modes=("mse" "mse_sam")
chuncks=()
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


for dataset in $datasets; do
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/mySOTA
    for concat in $concats; do
        for block_num in $block_nums; do
            for loss_mode in $loss_modes; do
                for base_model_name in $model_names; do

                    name_block_num=$(printf %02d $block_num)
                    model_name=$base_model_name\_$name_block_num\_$loss_mode\_$concat

                    # if [ -d ../SCI_result/$dataset\_sota/$model_name ]; then
                    python train_SOTA.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name
                    python evaluate_SOTA.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name
                    # fi

                    upload_model_name=$base_model_name\_$name_block_num\_$loss_mode\_$concat
                    mkdir -p ../SCI_result/$dataset\_$start_time/$upload_model_name/$upload_model_name\_upload
                    cp ../SCI_result/$dataset\_$start_time/$model_name/output.csv ../SCI_result/$dataset\_$start_time/$model_name/$upload_model_name\_upload/$upload_model_name\_output.csv
                    cp ../SCI_ckpt/$dataset\_$start_time/all_trained_sota/$model_name.tar ../SCI_result/$dataset\_$start_time/$model_name/$upload_model_name\_upload/$upload_model_name.tar
                    skicka upload ../SCI_result/$dataset\_$start_time/$upload_model_name/$upload_model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/mySOTA/$model_name
                    rm -rf ../SCI/result/$dataset\_$start_time/$upload_model_name
                done
            done
        done
    done
done

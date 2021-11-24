# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
# search_epoch=100
train_epoch=150
datasets=("CAVE" "Harvard")
model_names=("Attention" "HyperMix")
for i in {1..9}; do
    block_nums+=($i)
done
concats=('True')
loss_mode="mse"
start_time=$(date "+%m%d")
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
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/SOTA
    for concat in $concats; do
        for block_num in $block_nums; do
            for base_model_name in $model_names; do
                echo $dataset $concat $loss_mode $model_name

                name_block_num=$(printf %02d $block_num)
                model_name=$base_model_name\_$name_block_num\_$loss_mode\_$start_time\_$concat
                upload_model_name=$base_model_name\_$name_block_num\_$loss_mode\_$start_time\_$concat
                mkdir -p ../SCI_result/$dataset\_sota/$upload_model_name/$upload_model_name\_upload
                cp ../SCI_result/$dataset\_sota/$model_name/$model_name\_upload/$model_name\_output.csv ../SCI_result/$dataset\_sota/$upload_model_name/$upload_model_name\_upload/$upload_model_name\_output.csv
                cp ../SCI_result/$dataset\_sota/$model_name/$model_name\_upload/$model_name.tar ../SCI_result/$dataset\_sota/$upload_model_name/$upload_model_name\_upload/$upload_model_name.tar
                skicka upload ../SCI_result/$dataset\_sota/$upload_model_name/$upload_model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/SOTA/$model_name
                rm -rf ../SCI/result/$dataset\_sota/$upload_model_name
            done
        done
    done
done

# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
# search_epoch=100
train_epoch=150
datasets=("CAVE" "Harvard")
model_names=("Lambda")
block_num=9
concats=('False' 'True')
loss_modes=("mse" "mse_sam")
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
        for loss_mode in $loss_modes; do
            for model_name in $model_names; do
                echo $dataset $concat $loss_mode $model_name
                python reconst.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $model_name
                python refine.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $model_name
                python evaluate_lambda.py -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -m $model_name

                name_block_num=$(printf %02d $block_num)
                model_name=$model_name\_$name_block_num\_$loss_mode\_$start_time\_$concat
                reconst_name=Reconstruct_Stage\_$name_block_num\_$loss_mode\_$start_time\_$concat
                refine_name=Refine_Stage\_$name_block_num\_$loss_mode\_$start_time\_$concat
                mkdir -p ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
                cp ../SCI_result/$dataset\_$start_time/$model_name/output.csv ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/$model_name\_output.csv
                cp ../SCI_ckpt/$dataset\_$start_time/all_trained_sota/$reconst_name.tar  ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
                cp ../SCI_ckpt/$dataset\_$start_time/all_trained_sota/$refine_name.tar  ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
                skicka upload ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/SOTA/$model_name
            done
        done
    done
done

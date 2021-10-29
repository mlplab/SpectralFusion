# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
# search_epoch=100
train_epoch=150
datasets=("CAVE" "Harvard")
base_model_name="SpectralFusionRGBEncoder"
rgb_model_name="RGBEncoder"
block_nums=()
for i in {1..13}; do
    block_nums+=($i)
done
concats=('False')
modes=("inputOnly")
conv_mode="edsr"
edsr_modes=('normal' "separable")
loss_mode="mse"
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
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/$base_model_name
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/$rgb_model_name
    for block_num in $block_nums; do
        for concat in $concats; do
            for mode in $modes; do
                for edsr_mode in $edsr_modes; do

                    python train_rgb.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size  -md $mode -l $loss_mode -cm $conv_mode -em $edsr_mode
                    python evaluate_rgb.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -md $mode -l $loss_mode -cm $conv_mode -em $edsr_mode

                    name_block_num=$(printf %02d $block_num)
                    model_name=$rgb_model_name\_$name_block_num\_$loss_mode\_$mode\_$start_time\_$concat\_$conv_mode\_$edsr_mode
                    mkdir -p ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
                    cp ../SCI_result/$dataset\_$start_time/$model_name/output.csv ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/$model_name\_output.csv
                    cp ../SCI_ckpt/$dataset\_$start_time/all_trained/$model_name.tar ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/
                    skicka upload ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/$rgb_model_name/$model_name/


                    python train_rgb_fusion.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -md $mode -l $loss_mode -cm $conv_mode -em $edsr_mode
                    python evaluate_rgb_fusion.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -md $mode -l $loss_mode -cm $conv_mode -em $edsr_mode

                    name_block_num=$(printf %02d $block_num)
                    model_name=$base_model_name\_$name_block_num\_$loss_mode\_$mode\_$start_time\_$concat\_$conv_mode\_$edsr_mode
                    mkdir -p ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
                    cp ../SCI_result/$dataset\_$start_time/$model_name/output.csv ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/$model_name\_output.csv
                    cp ../SCI_ckpt/$dataset\_$start_time/all_trained/$model_name.tar ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/
                    skicka upload ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/$base_model_name/$model_name/

                done
            done
        done
    done
done

# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
# search_epoch=100
train_epoch=150
datasets=("ICVL")
# model_names=("Attention" "HyperMix")
model_names=("HyperMix")
# model_names=("HyperMix")
# block_nums=(5 7 9)
for i in {5..9..2}; do
    block_nums+=($i)
done
concats=('False')
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
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/mySOTA
    for concat in $concats; do
        for block_num in $block_nums; do
            for loss_mode in $loss_modes; do
                for base_model_name in $model_names; do
                    for chunck in $chuncks; do
                        end_status=1
                        end_count=0

                        name_block_num=$(printf %02d $block_num)
                        model_name=$base_model_name\_$name_block_num\_$loss_mode\_$concat\_$chunck
                        echo $model_name

                        if [ ! -e ../SCI_result/$dataset\_$start_time/$model_name/output.csv ]; then
                            while [ $end_status -eq 1 ]; do
                                python train_mix.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -ch $chunck
                                end_status=$?
                                let end_count++
                                if [ $end_count -eq 5 ]; then
                                    break
                                fi
                            done
                            if [ $end_count -eq 5 ]; then
                                callback_path=../SCI_ckpt/$dataset\_$start_time/$model_name\_callback/$model_name
                                latest_file=`ls -lt ${callback_path}/*.tar | head -n 1 | gawk '{print $9}'`
                                echo $latest_file
                                if [ -e $latest_file ]; then
                                    cp $latest_file ../SCI_ckpt/$dataset\_$start_time/all_trained_sota/$model_name.tar
                                else
                                    echo 'no callback path'
                                    exit 1
                                fi
                            fi
                            python evaluate_mix.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -ch $chunck

                            upload_model_name=$base_model_name\_$name_block_num\_$loss_mode\_$concat\_$chunck
                            mkdir -p ../SCI_result/$dataset\_$start_time/$upload_model_name/$upload_model_name\_upload
                            cp ../SCI_result/$dataset\_$start_time/$model_name/output.csv ../SCI_result/$dataset\_$start_time/$model_name/$upload_model_name\_upload/$upload_model_name\_output.csv
                            cp ../SCI_ckpt/$dataset\_$start_time/all_trained_sota/$model_name.tar ../SCI_result/$dataset\_$start_time/$model_name/$upload_model_name\_upload/$upload_model_name.tar
                            skicka upload ../SCI_result/$dataset\_$start_time/$upload_model_name/$upload_model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/mySOTA/$model_name
                            rm -rf ../SCI/result/$dataset\_$start_time/$upload_model_name
                            # done
                        fi
                    done
                done
            done
        done
    done
done

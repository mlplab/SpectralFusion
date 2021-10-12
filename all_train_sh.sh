# !/usr/bin/zsh


start_time=$(date "+%m%d")
train_epoch=150


while getopts b:e:s: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) train_epoch=$OPTARG ;;
        s) start_time=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


# zsh create_SOTA.sh -s $start_time
zsh create_fusion.sh -s $start_time -e $train_epoch
zsh create_hsiOnly.sh -s $start_time -e $train_epoch

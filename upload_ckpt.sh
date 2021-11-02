# !/usr/bin/zsh


CMDNAME=`basename $0`

start_time=$(date "+%m%d")
while getopts s: OPT
do
    echo "$OPTARG"
    case $OPT in
        s) start_time=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done

zsh upload_result_hsi.sh -s $start_time
zsh upload_result_fusion.sh -s $start_time
# zsh upload_result_SOTA.sh -s $start_time

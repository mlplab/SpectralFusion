# !/usr/bin/zsh


start_time=$(date "+%m%d")


zsh create_SOTA.sh -s $start_time
zsh create_fusion.sh -s $start_time
zsh create_hsiOnly.sh -s $start_time

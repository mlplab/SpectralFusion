# !/usr/bin/zsh


CMDNAME=`basename $0`

# flag='0'
# echo $?
# end_status=$?
# echo $?
# while [ $end_status -eq 0 ]
# do
#     echo unk
#     python test.py
#     end_status=$?
# done
# echo chinko
#
callback_path='../SCI_ckpt/ICVL_0119/Attention_07_mse_False_callback/Attention_07_mse_False'
latest_file=`ls -lt ${callback_path}/*.tar | head -n 1 | gawk '{print $9}'`
echo $latest_file

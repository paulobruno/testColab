nets=("lenet" "alexnet")
blocksize=("8" "10" "16" "25" "32" "50")

count_net=1
for net in ${nets[@]}
do
    echo "[ $count_net/${#nets[@]} ] Using $net architecture..."
    cp "networks/cnn_roi_"$net".py" "cnn_roi.py"
	count_bs=1
	for bs in ${blocksize[@]}
	do
	    echo "        [ $count_bs/${#blocksize[@]} ] Training with block size $bs..."
	    cp "settings/settings_"$bs".py" "settings.py"
	    python3 cnn_roi_train.py > "log_"$net"_"$bs".txt"
	    mv "cnn_roi_logs_train" "cnn_roi_logs_train_"$net"_"$bs
	    rm "settings.py"
    	((count_bs++))
	done
	rm "cnn_roi.py"
	((count_net++))
done

dd="sftsave_2b/"
ddd="checkpoint.epoch."
log="mem.txt"
fic="nancy_mem.txt"
for b in 0 1 2 3 4 5 6 7 8 9; do
	echo $dd$ddd$b >> $log
	python evaluatemem.py $dd/$ddd"$b" $fic | tee --append $log
done

log="gen.txt"
fic="testset.txt"
for b in 0 1 2 3 4 5 6 7 8 9; do
	echo $dd$ddd$b >> $log
	python evaluatemem.py $dd/$ddd"$b" $fic | tee --append $log
done

log="fgt.txt"
for b in 0 1 2 3 4 5 6 7 8 9; do
	echo $dd$ddd$b >> $log
	python ifeval2.py $dd/$ddd"$b" | tee --append $log
done 

exit

log="mem.txt"
mv $log lorasaves/
log="gen.txt"
mv $log lorasaves/
log="fgt.txt"
mv $log lorasaves/
mv qlora.log lorasaves/
mv lorasaves lorasaves_bs64_r32


log="mem.txt"
echo $a >> $log
python evaluatemem.py lorasaves/$a nancy_mem.txt | tee --append $log
a="checkpoint_epoch_1"
echo $a >> $log
python evaluatemem.py lorasaves/$a nancy_mem.txt | tee --append $log
a="checkpoint_epoch_2"
echo $a >> $log
python evaluatemem.py lorasaves/$a nancy_mem.txt | tee --append $log
a="checkpoint_epoch_3"
echo $a >> $log
python evaluatemem.py lorasaves/$a nancy_mem.txt | tee --append $log
a="checkpoint_epoch_4"
echo $a >> $log
python evaluatemem.py lorasaves/$a nancy_mem.txt | tee --append $log
exit

python ifeval2.py /home/xtof/results/.ep24
python evaluatemem.py /home/xtof/results/.ep0 nancy_mem.txt
python evaluatemem.py /home/xtof/results/.ep1 nancy_mem.txt
python evaluatemem.py /home/xtof/results/.ep2 nancy_mem.txt
python evaluatemem.py /home/xtof/results/.ep0 testset.txt
python evaluatemem.py /home/xtof/results/.ep1 testset.txt
python evaluatemem.py /home/xtof/results/.ep2 testset.txt


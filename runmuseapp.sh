#!/bin/bash
echo start 
rm test.csv

j=2.6
start=2.6
num2=3.0

touch test.csv
end="$(echo $start 0.1 | awk '{print $1 + $2; exit}')"
while (( $(echo "$end <= $num2" |bc -l) ))
do
 echo j is $j
 echo **************process is starting********************
 
 echo *****start is $start
 echo *****end is $end
 python3 preprocess.py $start $end
 b=0.05
 j=$( printf '%s + %s\n' "$j" "$b" | bc )


 start=$j
 end="$(echo $start 0.1| awk '{print $1 + $2; exit}')"
done 

python3 normalise.py





 






#!/bin/bash

red=$(tput setaf 1)
green=$(tput setaf 2)
blue=$(tput setaf 4)
normal=$(tput sgr0)

echo "Compiling binaries.."

echo $(make all)

echo "A long series of tests will run now.. Please wait!"


while read m n; do

    echo -e "\n------------- M="$m" N="$n", 3 iterations -------------"

    if [[ "$RUN_EXPERIMENTS_VB" = "1" ]]
    then
    	printf "${green}PART A: ${normal}\t| ${blue}PART B: ${normal}\t| ${red}PART C: ${normal} \n"
    fi

    sum1=0
    sum2=0
    sum3=0

    for iter in {1..3};
    do
        b1=$((./mainA "$m" "$n" | grep -iF "Elapsed" | awk '{print $6}') 2>&1)
        b2=$((./mainB "$m" "$n" | grep -iF "Elapsed" | awk '{print $6}') 2>&1)
        b3=$((./mainC "$m" "$n" | grep -iF "Elapsed" | awk '{print $6}') 2>&1)

	if [[ "$RUN_EXPERIMENTS_VB" = "1" ]]
	then
        	echo "$b1 msecs, $b2 msecs, $b3 msecs"
	fi

	sum1=$(echo $b1 + $sum1 | bc)
	sum2=$(echo $b2 + $sum2 | bc)
	sum3=$(echo $b3 + $sum3 | bc)

    done

    avg1=$(echo "scale=8; $sum1/3" | bc -l)
    avg2=$(echo "scale=8; $sum2/3" | bc -l)
    avg3=$(echo "scale=8; $sum3/3" | bc -l)

    printf "${green}Average ${normal}\t| ${blue}Average ${normal}\t| ${red}Average ${normal} \n"
    echo "$avg1 msecs, $avg2 msecs, $avg3 msecs"

done < expsizes


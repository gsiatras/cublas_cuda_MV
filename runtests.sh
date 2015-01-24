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

    printf "${green}PART A: ${normal}\t| ${blue}PART B: ${normal}\t| ${red}PART C: ${normal} \n"
    
    for iter in {1..3};
    do
        b1=$((./mainA "$m" "$n" | grep -iF "Elapsed" | awk '{print $6 " " $7}') 2>&1)
        b2=$((./mainB "$m" "$n" | grep -iF "Elapsed" | awk '{print $6 " " $7}') 2>&1)
        b3=$((./mainC "$m" "$n" | grep -iF "Elapsed" | awk '{print $6 " " $7}') 2>&1)
        echo "$b1, $b2, $b3"
    done

done < expsizes


#!/bin/bash

red=$(tput setaf 1)
green=$(tput setaf 2)
blue=$(tput setaf 4)
normal=$(tput sgr0)

p1=~/partA/
p2=~/partB/
p3=~/partC/

echo "Compiling binaries.."

echo $(cd "$p1" && ./makeit)
echo $(cd "$p2" && ./makeit)
echo $(cd "$p3" && ./makeit)

echo "A long series of tests will run now.. Please wait!"

while read m n; do

    echo -e "\n------------- M="$m" N="$n", 3 iterations -------------"

	#exp[0]=$((cd "$p1" && ./makeit && ./main "$m" "$n") 2>&1)
	#exp[1]=$((cd "$p2" && ./makeit && ./main "$m" "$n") 2>&1)
	#exp[2]=$((cd "$p3" && ./makeit && ./main "$m" "$n") 2>&1)

    printf "${green}PART A: ${normal} | ${blue}PART B: ${normal} | ${red}PART C: ${normal}"
    
    for iter in {1..3};
    do
        exp[0]=$((./main "$m" "$n") 2>&1)
        exp[1]=$((./main "$m" "$n") 2>&1)
        exp[2]=$((./main "$m" "$n") 2>&1)

        printf "%s, "  "${exp[0]}" | grep -iF "Elapsed" | awk '{print $5 " " $6}'
        # printf "${blue}PART B: ${normal}" 
        printf "%s, " "${exp[1]}" | grep -iF "Elapsed" | awk '{print $5 " " $6}'
        # printf "${red}PART C: ${normal}"
        printf "%s"  "${exp[2]}" | grep -iF "Elapsed" | awk '{print $5 " " $6}'
    done

done < expsizes


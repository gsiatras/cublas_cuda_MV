#!/bin/bash

red=$(tput setaf 1)
green=$(tput setaf 2)
blue=$(tput setaf 4)
normal=$(tput sgr0)

p1=~/partA/
p2=~/partB/
p3=~/partC/

echo "Please wait while running tests.."

while read n m; do

	echo "------------- " "$n" "$m"

	exp[0]=$((cd "$p1" && ./makeit && ./main "$n" "$m") 2>&1)
	exp[1]=$((cd "$p2" && ./makeit && ./main "$n" "$m") 2>&1)
	exp[2]=$((cd "$p3" && ./makeit && ./main "$n" "$m") 2>&1)

	printf "${green}PART A: ${normal}"
	printf "%s \n"  "${exp[0]}" | grep "Elapsed"
	printf "${blue}PART B: ${normal}" 
	printf "%s \n" "${exp[1]}" | grep "Elapsed"
	printf "${red}PART C: ${normal}"
	printf "%s \n"  "${exp[2]}" | grep "Elapsed"

done < testsizes


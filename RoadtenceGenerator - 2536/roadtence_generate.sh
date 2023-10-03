#!/bin/bash

readarray -t lines < ./region_name_temp.txt


for ((i=0; i<${#lines[@]}; i++));
do
	trimed_region=$(echo "${lines[$i*2]}" | tr -d ' ')
	python3 "RoadtenceGenerator - expand distance.py" "${trimed_region}" > ${trimed_region}_roadtence_generate_log.txt
done

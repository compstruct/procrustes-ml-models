
#$1 #=1000000

#scancel  $(squeueMe | grep $1 | sed 's/\s\s*/,/g' | cut -d',' -f2 | tr '\n' ' ' | tail -1) | sed 's/.*+//g'



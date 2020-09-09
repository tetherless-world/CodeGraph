set terminal pdf

set xlabel "Number of results requested"
set ylabel "Mean result rank in title text search"

set key outside top

plot "ranks.dat" using 1:2 with lines title "USE titles", "ranks.dat" using 1:3 with lines title "USE all", "ranks.dat" using 1:4 with lines title "Bert titles", "ranks.dat" using 1:5 with lines title "Bert all", "ranks.dat" using 1:6 with lines title "Roberta titles", "ranks.dat" using 1:7 with lines title "Roberta all", "ranks.dat" using 1:8 with lines title "Search all"


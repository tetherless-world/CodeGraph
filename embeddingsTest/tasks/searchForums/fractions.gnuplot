set terminal pdf

set xlabel "Number of results requested"
set ylabel "Overlap with title text search"

set key outside top

plot "fractions.dat" using 1:2 with lines title "USE titles", "fractions.dat" using 1:3 with lines title "USE all", "fractions.dat" using 1:4 with lines title "Bert titles", "fractions.dat" using 1:5 with lines title "Bert all", "fractions.dat" using 1:6 with lines title "Roberta titles", "fractions.dat" using 1:7 with lines title "Roberta all", "fractions.dat" using 1:8 with lines title "Search all"


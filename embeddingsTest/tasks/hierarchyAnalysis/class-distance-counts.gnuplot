set terminal pdf

set key off

set xrange [0:11]

set boxwidth 0.9 relative
set style fill solid 1.0

plot 'class-distance-counts.dat' with boxes

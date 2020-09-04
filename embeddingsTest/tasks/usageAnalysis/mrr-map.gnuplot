set terminal pdf
fr(x) = a + b*x
fit fr(x) 'mrr-map.dat' u 1:2 via a,b
fp(x) = c + d*x
fit fp(x) 'mrr-map.dat' u 1:3 via c,d
set xlabel "number of shared call sequences"
plot "mrr-map.dat" using 1:2 with lines title "Mean Reciprocal Rank (MRR)", fr(x) title "MRR trend", "mrr-map.dat" using 1:3 with lines title "Mean Average Precision (MAP)", fp(x) title "MAP trend"
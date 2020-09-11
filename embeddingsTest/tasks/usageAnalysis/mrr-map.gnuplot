set terminal pdf
set xlabel "minimum number of shared call sequences"
set xrange [2:105]

set key outside top

set pointsize .2

fr(x) = a + b*x
fit fr(x) 'mrr-map-use.dat' u 1:2 via a,b
fp(x) = c + d*x
fit fp(x) 'mrr-map-use.dat' u 1:4 via c,d

frb(x) = e + f*x
fit frb(x) 'mrr-map-bert.dat' u 1:2 via e,f
fpb(x) = g + h*x
fit fpb(x) 'mrr-map-bert.dat' u 1:4 via g,h

frr(x) = i + j*x
fit frr(x) 'mrr-map-roberta.dat' u 1:2 via i,j
fpr(x) = k + l*x
fit fpr(x) 'mrr-map-roberta.dat' u 1:4 via k,l

plot "mrr-map-use.dat" using 1:2 lc 1 pt 2 title "MRR\tUSE", fr(x) lc 1 title "MRR trend\tUSE", "mrr-map-use.dat" using 1:4 lc 2 pt 2 title "MAP\tUSE", fp(x) lc 2 title "MAP trend\tUSE", "mrr-map-bert.dat" using 1:2 lc 3 pt 2 title "MRR\tBert", frb(x) lc 3 title "MRR trend\tBert", "mrr-map-bert.dat" using 1:4 lc 4 pt 2 title "MAP\tBert", fpb(x) lc 4 title "MAP trend\tBert", "mrr-map-roberta.dat" using 1:2 lc 5 pt 2 title "MRR\tRoberta", frr(x) lc 5 title "MRR trend\tRoberta", "mrr-map-roberta.dat" using 1:4 lc 6 pt 2 title "MAP\tRoberta", fpr(x) lc 6 title "MAP trend\tRoberta"
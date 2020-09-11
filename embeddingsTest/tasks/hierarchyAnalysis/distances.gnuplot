set terminal pdf

set logscale y

set border 3

set ylabel "Number of docstring pairs"
set xlabel "Distance by USE embeddings"

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

# Each bar is half the (visual) width of its x-range.
set boxwidth 0.01 absolute
set style fill solid 1.0 noborder

bin_width = 0.1;

bin_number(x) = floor(x/bin_width)

rounded(x, o) = bin_width * ( bin_number(x) + o )

plot 'results/1_distance.txt' using (rounded($1, 0)):(1) smooth frequency with lines title "1", 'results/2_distance.txt' using (rounded($1, .1)):(1) smooth frequency with lines title "2", 'results/3_distance.txt' using (rounded($1, .2)):(1) smooth frequency with lines title "3", 'results/4_distance.txt' using (rounded($1, .3)):(1) smooth frequency with lines title "4", 'results/5_distance.txt' using (rounded($1, .4)):(1) smooth frequency with lines title "5", 'results/6_distance.txt' using (rounded($1, .5)):(1) smooth frequency with lines title "6", 'results/7_distance.txt' using (rounded($1, .6)):(1) smooth frequency with lines title "7", 'results/8_distance.txt' using (rounded($1, .7)):(1) smooth frequency with lines title "8", 'results/9_distance.txt' using (rounded($1, .8)):(1) smooth frequency with lines lc "pink" title "9", 'results/10_distance.txt' using (rounded($1, .9)):(1) smooth frequency with lines lc "cyan" title "10"

set terminal pdf

set border 3

set ylabel "Number of post pairs"
set xlabel "Distance by USE embeddings"

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

# Each bar is half the (visual) width of its x-range.
set boxwidth 0.1 absolute
set style fill solid 1.0 noborder

bin_width = 0.1;

bin_number(x) = floor(x/bin_width)

rounded(x) = bin_width * ( bin_number(x) )

plot 'true.dat' using (rounded($1)):(1) smooth frequency with lines lc 1 title "Linked posts", 'false.dat' using (rounded($1)):(1) smooth frequency with lines lc 1 dt 2 title "Non-linked posts"

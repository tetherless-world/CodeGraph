set terminal pdf

set border 3

set ylabel "Number of post pairs"
set xlabel "Distance by embeddings"

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

# Each bar is half the (visual) width of its x-range.
set boxwidth 0.05 absolute
set style fill solid 1.0 noborder

bin_width = 0.2;

bin_number(x) = floor(x/bin_width)

rounded(x, o) = bin_width * ( bin_number(x) + o )

plot 'true.dat' using (rounded($1, .75)):(1) smooth frequency with boxes title "Linked posts", 'false.dat' using (rounded($1, 1)):(1) smooth frequency with boxes title "Non-linked posts"

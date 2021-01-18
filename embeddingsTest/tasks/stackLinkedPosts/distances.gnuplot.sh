GP_FILE=$1
shift

cat > $GP_FILE <<EOF
set terminal pdf

set key under

set border 3

set ylabel "Number of post pairs"
set xlabel "Embedding distance"

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

# Each bar is half the (visual) width of its x-range.
set boxwidth 0.1 absolute
set style fill solid 1.0 noborder

bin_width = 0.1;

bin_number(x) = floor(x/bin_width)

rounded(x) = bin_width * ( bin_number(x) )
EOF

i=1
sep=""
echo -n "plot" >> $GP_FILE
for stem in $@; do
    echo -n "$sep '$stem.1' using (rounded(\$1)):(1) smooth frequency with lines lc $i title \"Linked $stem\", '$stem.2' using (rounded(\$1)):(1) smooth frequency with lines lc $i dt 2 title \"Non-linked $stem\"" >> $GP_FILE
    i=`expr $i + 1`
    sep=","
done

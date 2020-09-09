BEGIN {
    col = 1
    delete tab
}

{
    for(i = 1; i <= NF; i++) {
	tab[i][col] = $i
    }

    row = i
    col++
}

END {
    for(j = 1; j < row; j++) {
	for(k = 1; k <= col; k++) {
	    printf("%s\t", tab[j][k])
	}
	print("")
    }
}

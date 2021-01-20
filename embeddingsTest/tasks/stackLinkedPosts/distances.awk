BEGIN {
    max = -1000000
    min = 1000000
    delete counts
    file = 0
}

FNR==1 {
    file++
    counts[file] = 0
}

/[0-9.]+/ {
    match($0, /([0-9.]+)/, a)

    if (a[1] < min) {
	min = a[1]
    }
    
    if (a[1] > max) {
	max = a[1]
    }
    
    data[counts[file], file] = a[1]
    counts[file]++
}

END {
    print file
    for(i = 0; i <= file; i++) {
	for(r = 0; r < counts[i]; r++) {
#	    print((data[r, i] - min) / (max - min)) >dataFiles "." i
	    print(data[r, i]) >dataFiles "." i
	}
    }
}

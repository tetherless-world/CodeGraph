/^[0-9]+: mrr: [0-9.]+$/ {
    delete a;
    match($0, /(^[0-9]+): mrr: ([0-9.]+)$/, a);
    mrr[a[1]] = a[2];
}

/^[0-9]+: mrrse: [0-9.]+$/ {
    delete a;
    match($0, /(^[0-9]+): mrrse: ([0-9.]+)$/, a);
    mrrse[a[1]] = a[2];
}

/^[0-9]+: map@10: [0-9.]+$/ {
    delete a;
    match($0, /(^[0-9]+): map@10: ([0-9.]+)$/, a);
    map[a[1]] = a[2];
}

/^[0-9]+: mapse: [0-9.]+$/ {
    delete a;
    match($0, /(^[0-9]+): mapse: ([0-9.]+)$/, a);
    mapse[a[1]] = a[2];
}

END {
    for(i = start; i > 0; i--) {
	if (i in mrr && i in map)  {
	    print(i "\t" mrr[i] "\t" mrrse[i] "\t" map[i] "\t" mapse[i]);
	}
    }
}

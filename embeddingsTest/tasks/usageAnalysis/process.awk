{
    patsplit($0, a, /[, \[\]]+/, b)
    for(s in b) {
	print(b[s]);
    }
}

int _error_handler(const char * _error) {
	fprintf(stderr, "%s\n", _error);
	exit(EXIT_FAILURE);
}

double randDouble() {
	double rn;
	int rmin = -100000, rmax = 100000;

	rn = (double) rand() / RAND_MAX;
	rn = rmin + rn * (rmax - rmin);

	return rn;
}
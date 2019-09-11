import ParseDocs


def test1():

    s = "Returns the mean accuracy on the given test data and labels.\n\nIn multi-label classification, this \
    is the subset accuracy\nwhich is a harsh metric since you require for each sample that\neach label set be correctly predicte\
    d.\n\nParameters\n----------\nX : array-like, shape = (n_samples, n_features)\n    Test samples.\n\ny : array-like, shape = \
    (n_samples) or (n_samples, n_outputs)\n    True labels for X.\n\nsample_weight : array-like, shape = [n_samples], optional\n\
        Sample weights.\n\nReturns\n-------\nscore : float\n    Mean accuracy of self.predict(X) wrt. y."

    do_test(s)


def do_test(s, param_names):
    methoddoc, paramdoc, returndoc = ParseDocs.getDocStructure(s)
    print('method')
    print(methoddoc)
    print('params')
    print(paramdoc)
    print('returns')
    print(returndoc)
    print(ParseDocs.create_parameter_map(paramdoc, param_names))


def test2():
    s = "Performs clustering on X and returns cluster labels.\n\nParameters\n----------\nX : ndarray, shape (n_samples, n_features)\n    Input data.\n\nReturns\n-------\ny : ndarray, shape (n_samples,)\n    cluster labels"
    do_test(s)

def test3():
    s = "Row and column indices of the i'th bicluster.\n\nOnly works if ``rows_`` and ``columns_`` attributes\
 exist.\n\nParameters\n----------\ni : int\n    The index of the cluster.\n\nReturns\n-------\nrow_ind : np.array, dtype=np.\
intp\n    Indices of rows in the dataset that belong to the bicluster.\ncol_ind : np.array, dtype=np.intp\n    Indices of co\
lumns in the dataset that belong to the bicluster."
    do_test(s)

def test4():
    s ="Predict using the linear model\n\nParameters\n----------\nX : {array-like, sparse matrix}, shape = (n_\
samples, n_features)\n    Samples.\n\nReturns\n-------\nC : array, shape = (n_samples,)\n    Returns predicted values."
    do_test(s)

def test5():
    s = 'Returns the k-th diagonal of the matrix. \n \
        Parameters \n \
        ---------- \n \
        k : int, optional \n \
            Which diagonal to set, corresponding to elements a[i, i+k]. \n \
            Default: 0 (the main diagonal). \n \
            .. versionadded:: 1.0 \n \
        See also \n \
        -------- \n \
        numpy.diagonal : Equivalent numpy function. \n \
        Examples \n \
        -------- \n \
        >>> from scipy.sparse import csr_matrix \n \
        >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]) \n \
        >>> A.diagonal() \n \
        array([1, 0, 5]) \n \
        >>> A.diagonal(k=1) \n \
        array([2, 3])'
    do_test(s)

def test6():
    s = "Compute the arithmetic mean along the specified axis.\n\nReturns the average of the matrix elements. The average is taken\nover all elements in the matrix by default, otherwise over the\nspecified axis. `float64` intermediate and return values are used\nfor integer inputs.\n\nParameters\n----------\naxis : {-2, -1, 0, 1, None} optional\n    Axis along which the mean is computed. The default is to compute\n    the mean of all elements in the matrix (i.e. `axis` = `None`).\ndtype : data-type, optional\n    Type to use in computing the mean. For integer inputs, the default\n    is `float64`; for floating point inputs, it is the same as the\n    input dtype.\n\n    .. versionadded:: 0.18.0\n\nout : np.matrix, optional\n    Alternative output matrix in which to place the result. It must\n    have the same shape as the expected output, but the type of the\n    output values will be cast if necessary.\n\n    .. versionadded:: 0.18.0\n\nReturns\n-------\nm : np.matrix\n\nSee Also\n--------\nnp.matrix.mean : NumPy's implementation of 'mean' for matrices"
    do_test(s)


def test7():
    s = "reshape(self, shape, order='C', copy=False)\n\nGives a new shape to a sparse matrix without changing its data.\n\nParameters\n----------\nshape : length-2 tuple of ints\n    The new shape should be compatible with the original shape.\norder : {'C', 'F'}, optional\n    Read the elements using this index order. 'C' means to read and\n    write the elements using C-like index order; e.g. read entire first\n    row, then second row, etc. 'F' means to read and write the elements\n    using Fortran-like index order; e.g. read entire first column, then\n    second column, etc.\ncopy : bool, optional\n    Indicates whether or not attributes of self should be copied\n    whenever possible. The degree to which attributes are copied varies\n    depending on the type of sparse matrix being used.\n\nReturns\n-------\nreshaped_matrix : sparse matrix\n    A sparse matrix with the given `shape`, not necessarily of the same\n    format as the current object.\n\nSee Also\n--------\nnp.matrix.reshape : NumPy's implementation of 'reshape' for matrices"
    do_test(s)

def test8():
    s = "Fit linear model. Parameters ---------- X : numpy array or sparse matrix of shape [n_samples,n_features] Training data y : numpy array of shape [n_samples, n_targets] Target values. Will be cast to X's dtype if necessary sample_weight : numpy array of shape [n_samples] Individual weights for each sample .. versionadded:: 0.17 parameter *sample_weight* support to LinearRegression. Returns ------- self : returns an instance of self."
    do_test(s, ['X', 'y', 'sample_weights'])

test8()
test7()
test6()
test5()
test1()
test2()
test3()
test4()


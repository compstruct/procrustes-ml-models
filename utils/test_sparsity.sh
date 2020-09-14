
python -c 'import numpy;import sys;array_path=sys.argv[1];A = numpy.load(array_path);sparsity=1-sum(sum(sum(sum(sum(A != 0 )))))/ numpy.size(A);print("\t| Sparsity= %0.2f" % (sparsity));print("| Shape=",numpy.shape(A));print("| Total=",numpy.size(A))' $1 


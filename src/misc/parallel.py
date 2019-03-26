try:
    from joblib import Parallel, delayed
except Exception as e:
    print(e) #TODO
    # if joblib does not exist just run it in a single thread
    delayed = lambda x: x
    def Parallel( *args, **kwargs ):
        return list

# Allow pickling member functions
def _pickle_method(method):
    func_name = method.__name__
    obj = method.__self__
    return _unpickle_method, (func_name, obj)

def _unpickle_method(func_name, obj):
    try:
        return obj.__getattribute__(func_name)
    except AttributeError:
        return None

#The copyreg module offers a way to define functions used while pickling specific objects
# Requires to install future module
import types
try:
    import copyreg
except:
    import six.moves.copyreg as copyreg

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
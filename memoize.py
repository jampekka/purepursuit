import inspect
import functools
import pickle
import hashlib
from pathlib import Path

class DirShelve:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)

    def __contains__(self, key):
        return (self.directory / key).exists()

    def __getitem__(self, key):
        return pickle.load(open(self.directory / key, 'rb'))
    
    def __setitem__(self, key, value):
        return pickle.dump(value, open(self.directory / key, 'wb'))

storage = DirShelve("CACHEDIR")

objhash = lambda x: hashlib.md5(pickle.dumps(x)).hexdigest()

def memoize(f):
    func_code = inspect.getsource(f)
    func_hash = objhash(func_code)
    func_name = f.__name__
    
    @functools.wraps(f)
    def callit(*args, **kwargs):
        callspec = func_name, func_hash, objhash((args, kwargs))
        callspec = '.'.join(callspec)
        if callspec in storage:
            return storage[callspec]
        
        result = f(*args, **kwargs)
        storage[callspec] = result

        return result

    return callit

def unusedfunc(): pass

def outerfunc():
    return 3

def test():
    @memoize
    def justonce(wtf):
        print("Called")
        return wtf*outerfunc()

    print(justonce(2))
    print(justonce(2))

if __name__ == '__main__': test()




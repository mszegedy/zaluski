# zaluski/filedatabuffer.py
# by Maria Szegedy, 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''Provides an object, FileDataBuffer, that robustly turns the costly
computation of derived properties of files into essentially a key-value store,
caching them in the process.'''

### Imports
## Python Standard Library
import re
import collections, types, copy, enum
import itertools
import zlib
import hashlib
import os, io, time
import json, base64
import warnings, inspect
import fcntl
from functools import reduce

## Other stuff
import sortedcontainers
import dill
import numpy as np
from mpi4py import MPI

### Constants

DEBUG = False

MPICOMM = MPI.COMM_WORLD
MPIRANK = MPICOMM.Get_rank()
MPISIZE = MPICOMM.Get_size()
MPISTATUS = MPI.Status()
MPI.pickle = dill # upgrade

### Debug functions

def _DEBUG_OUT(*args, **kwargs):
    '''Print something, but only if DEBUG is True. Also, always flush output.'''
    if DEBUG:
        print('DEBUG: ', end='')
        kwargs['flush'] = True
        print(*args, **kwargs)
    try:
        return args[0]
    except IndexError:
        pass
_DEBUG_OUT('Debug is on.')

### Decorators

def returns_ndarray(f):
    '''Sets an attribute on the function that tells FileDataBuffer to make
    import_ and export_ methods for this function that respectively decompress
    and compress the Numpy array it returns. Useless for functions that aren't
    calculate_ methods in FileDataBuffer.'''
    f.returns_ndarray = True
    return f

def returns_pickleable(f):
    '''Sets an attribute on the function that tells FileDataBuffer to make
    import_ and export_ methods for this function that respectively decompress
    and compress the pickleable object it returns, using dill. Useless for
    functions that arent calculate_ methods in FileDataBuffer.'''
    f.returns_pickleable = True
    return f

def non_cacheable(f):
    '''Sets an attribute on the function that tells FileDataBuffer to not even
    attempt to cache the result of a calculate_ method, or store it in any way
    (not even in self.data, except temporarily for gather_ methods).
    @returns_ndarray is redundant with this, because FileDataBuffer doesn't care
    about the types of non-cached return values. For use with calculate_ methods
    that return enormous amounts of data and yet are easily computable.'''
    f.non_cacheable = True
    return f

### Special compress/decompress functions

def compress_ndarray(array):
    '''Compress a numpy ndarray into a JSON-serializable object.'''
    # Final form is:
    # [bunch of bits as Base 85-encoded string,
    #  tuple containing matrix side lengths]
    # this 7 is the compression level --------------------- V
    return [base64.b85encode(zlib.compress(array.tobytes(), 7))
                  .decode(encoding='utf-8'),
            array.shape,
            str(array.dtype)]

def decompress_ndarray(compressed_array):
    '''Recover a numpy ndarray compressed by compress_ndarray().'''
    data, shape, dtype = compressed_array
    return np.reshape(
        np.frombuffer(
            zlib.decompress(
                base64.b85decode(
                    data.encode(encoding='utf-8'))),
            np.dtype(dtype))[:reduce(lambda x, y: x*y, shape)],
        shape)

def compress_pickleable(o):
    '''Compress anything that dill can dump into a JSON-serializable object.'''
    # Converts to a Base 85-encoded string
    return base64.b85encode(dill.dumps(o))

def decompress_pickleable(compressed_o):
    '''Recover an object compressed by compress_pickleable().'''
    return dill.loads(base64.b85decode(compressed_o))

### FileDataBuffer helper classes

class BaseTypes(enum.Enum):
    '''Represents a classification of the type of an argument, or if the
    argument is a collection, a classification of the types of the things in
    said collection. We currently only care about whether it's a string
    containing a path to an existing file (PATH) or something else (MISC).'''
    MISC = 'MISC'
    PATH = 'PATH'

class ContainerTypes(enum.Enum):
    '''Represents a classification of the type of an argument when it's a
    container. We currently only care about whether it's a set (SET), a sequence
    (SEQUENCE), or something else (MISC).'''
    MISC = 'MISC'
    SET = 'SET'
    SEQUENCE = 'SEQUENCE'

class _Argument:
    '''The object that _ProtoArgs uses to represent a single argument.'''
    def __init__(self, value, file_paths_dict):
        def encapsulate(path):
            return EncapsulatedFile(
                path,
                file_paths_dict.setdefault(os.path.abspath(path), {}))

        if isinstance(value, str):
            try:
                _ = open(value)
                self.value = encapsulate(value)
                self.base_type = BaseTypes.PATH
                self.container_type = ContainerTypes.MISC
                return
            except FileNotFoundError:
                pass
        if isinstance(value, (collections.abc.Set, collections.abc.Sequence)) \
           and all(isinstance(item, str) for item in value):
            try:
                _ = tuple(open(item) for item in value)
                self.value = type(value)(encapsulate(path) for path in value)
                self.base_type = BaseTypes.PATH
                if isinstance(value, collections.abc.Set):
                    self.container_type = ContainerTypes.SET
                else:
                    self.container_type = ContainerTypes.SEQUENCE
                return
            except FileNotFoundError:
                pass
        self.value          = value
        self.base_type      = BaseTypes.MISC
        self.container_type = ContainerTypes.MISC
    @property
    def accessor_items(self):
        '''Items to add to the accessor tuple for this argument.'''
        if self.base_type == BaseTypes.MISC:
            yield self.value
            return
        if self.container_type == ContainerTypes.MISC:
            yield self.value.hash
            return
        if self.container_type == ContainerTypes.SEQUENCE:
            for f in self.value:
                yield f.hash
            return
        if self.container_type == ContainerTypes.SET:
            # sorted by hash (see EncapsulatedFile)
            for f in sortedcontainers.SortedSet(self.value):
                yield f.hash
            return
    @property
    def encapsulated_files(self):
        '''Encapsulated files to add to self_.encapsulated_files for this
        argument.'''
        if self.base_type == BaseTypes.MISC:
            return
        if self.container_type == ContainerTypes.MISC:
            yield self.value
            return
        if self.container_type == ContainerTypes.SEQUENCE:
            for f in self.value:
                yield f
            return
        if self.container_type == ContainerTypes.SET:
            # sorted by hash (see EncapsulatedFile)
            for f in sortedcontainers.SortedSet(self.value):
                yield f
            return
    @property
    def calcvalue(self):
        '''Value of arg for calling calculate_ with.'''
        if self.base_type == BaseTypes.MISC:
            return self.value
        if self.container_type == ContainerTypes.MISC:
            return self.value.path
        return type(self.value)(f.path for f in self.value)

class _ProtoArgs():
    '''An object that get_ and gather_ methods use to construct arguments to
    call their associated calculate_ methods with. More information under
    FileDataBuffer._proto_args().'''
    def __init__(self, calcfxn, file_paths_dict, args, kwargs):
        # index of each argument (with number indices for positional
        # arguments, but string indices for keyword arguments); needed
        # to provide iteration (unused)
        self.indices = []
        # positional args and their characteristics; accessed through
        # property .args
        self._args = []
        # keyword args and their characteristics; accessed through
        # property .kwargs
        self._kwargs = {}
        # list of encapsulated files for file-type args; needed to
        # generate the three properties .paths, .prime_path, and
        # .prime_hash
        self.encapsulated_files = []
        # the string that can be used as a key to index into the parent
        # FileDataBuffer's .data dict to retrieve the information
        # stored by this computation:
        self.accessor_string = ''
        argspec = inspect.getfullargspec(calcfxn)
        accessor_list = []
        lendefaults = len(argspec.defaults) \
                      if argspec.defaults is not None \
                      else 0
        # start from 1 because we skip over self
        if lendefaults > 0:
            positargs = argspec.args[1:-lendefaults]
        else:
            positargs = argspec.args[1:]
        for i in range(len(positargs)):
            self.indices.append(i)
            arg = _Argument(args[i], file_paths_dict)
            self._args.append(arg)
            self.encapsulated_files.extend(arg.encapsulated_files)
            accessor_list.extend(arg.accessor_items)
        if lendefaults > 0:
            namedargs = (argspec.args[-lendefaults:] + \
                         argspec.kwonlyargs)
        else:
            namedargs = argspec.kwonlyargs
        namedargs = (kwarg for kwarg in namedargs \
                     if kwarg in kwargs.keys())
        for kwarg_name in namedargs:
            self.indices.append(kwarg_name)
            arg = _Argument(kwargs[kwarg_name], file_paths_dict)
            self._kwargs[kwarg_name] = arg
            self.encapsulated_files.extend(arg.encapsulated_files)
            accessor_list.extend(arg.accessor_items)
        self.accessor_string = str(tuple(accessor_list))
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.args[index]
        return self.kwargs[index]
    def __iter__(self):
        nargs = len(self.args)
        return itertools.chain((self.args[i] for i in range(nargs)),
                               (self.kwargs[i] \
                                for i in self.indices[nargs:]))
    @property
    def args(self):
        '''This _ProtoArgs's represented positional args.'''
        return [arg.value for arg in self._args]
    @property
    def kwargs(self):
        '''This _ProtoArgs's represented keyword args.'''
        return {name: arg.value for name, arg in self._kwargs.items()}
    @property
    def paths(self):
        '''Paths of encapsulated files. Needed by parent FileDataBuffer
        to know where to retrieve caches from before computation is
        performed, and where to update them afterwards. Also forms basis
        of .prime_path.'''
        return [encapsulated.path \
                for encapsulated in self.encapsulated_files]
    @property
    def hashes(self):
        '''Unused, except for defining .prime_hash. For symmetry with
        .paths.'''
        return [encapsulated.hash \
                for encapsulated in self.encapsulated_files]
    @property
    def prime_path(self):
        '''The path where the data from whatever computation these args
        will be used for will be stored.'''
        return self.paths[0]
    @property
    def prime_hash(self):
        '''The hash of the file the computed data "belongs" to.'''
        return self.hashes[0]
    @property
    def calcargs(self):
        '''The positional arguments used for a calculate_ call.'''
        return [arg.calcvalue for arg in self._args]
    @property
    def calckwargs(self):
        '''The keyword arguments used for a calculate_ call.'''
        return {name: arg.calcvalue \
                for name, arg in self._kwargs.items()}
    def update_paths(self):
        '''Make the master FileDataBuffer of this _Argument's
        EncapsulatedFiles aware of the respective associated files' mtimes and
        hashes.'''
        for f in self.encapsulated_files:
            f.update_file_paths_dict()

class EncapsulatedFile:
    '''The class responsible for handling access to the paths that are passed to
    a FileDataBuffer, and for keeping track of when they change.'''
    def __init__(self, path, file_paths_dict):
        if path is None:
            self.path = None
            self.file_path_dict = {}
            self.mtime = 0
            self.hash = '-1' # needs to be an integer for comparisons
            self._stream = None
        else:
            self.path = os.path.abspath(path)
            self.file_paths_dict = file_paths_dict
            self.mtime = self.file_paths_dict.setdefault('mtime', 0)
            self.hash = self.file_paths_dict.setdefault('hash', None)
            diskmtime = os.path.getmtime(self.path)
            if diskmtime > self.mtime or self.hash is None:
                self.update(diskmtime)
    def __lt__(self, other):
        return self.__compare(other, lambda s, o: s < o)
    def __le__(self, other):
        return self.__compare(other, lambda s, o: s <= o)
    def __eq__(self, other):
        return self.__compare(other, lambda s, o: s == o)
    def __ge__(self, other):
        return self.__compare(other, lambda s, o: s >= o)
    def __gt__(self, other):
        return self.__compare(other, lambda s, o: s > o)
    def __ne__(self, other):
        return self.__compare(other, lambda s, o: s != o)
    def __hash__(self):
        return int(self.hash, 16)
    def __compare(self, other, method):
        try:
            return method(hash(self), hash(other))
        except (AttributeError, TypeError):
            return NotImplemented
    def update_file_paths_dict(self):
        '''Make this EncapsulatedFile's master FileDataBuffer aware of the mtime
        and hash of our associated file.'''
        if self.path is not None:
            self.file_paths_dict['mtime'] = self.mtime
            self.file_paths_dict['hash'] = self.hash
    def update(self, mtime=None):
        if self.path is not None:
            diskmtime = mtime or os.path.getmtime(self.path)
            if diskmtime > self.mtime or self._stream is None:
                with open(self.path, 'rb') as pdb_file:
                    contents = pdb_file.read()
                self.mtime = diskmtime
                hash_fun = hashlib.md5()
                hash_fun.update(contents)
                self.hash = hash_fun.hexdigest()
                self.update_file_paths_dict()

### FileDataBuffer and friends

def _make_get(data_name):
    '''Create a get_ accessor function for a FileDataBuffer data type.'''
    # Python duck typing philosophy says we shouldn't do any explicit checking
    # of the thing stored at the attribute, but in any case, it should be a
    # function that takes at least one file path, with the rest of its
    # arguments being hashables. See the FileDataBuffer docs for more info.
    def get(self_, *args, **kwargs):
        calculate = getattr(self_, 'calculate_'+data_name)
        proto_args = self_._proto_args(calculate, args, kwargs)
        if hasattr(calculate, 'non_cacheable'):
            return calculate(*proto_args.calcargs,
                             **proto_args.calckwargs)
        for path in proto_args.paths:
            self_.retrieve_data_from_cache(os.path.dirname(path))
        file_data = self_.data \
                         .setdefault(proto_args.prime_hash, {}) \
                         .setdefault(data_name, {})
        try:
            if hasattr(self_, 'import_'+data_name):
                return getattr(self_, 'import_'+data_name) \
                              (file_data[proto_args.accessor_string])
            return file_data[proto_args.accessor_string]
        except KeyError:
            if not self_.calculatingp:
                raise KeyError('That file\'s not in the cache.')
            # construct new args
            if hasattr(self_, 'export_'+data_name):
                file_data[proto_args.accessor_string] = \
                    getattr(self_, 'export_'+data_name) \
                           (calculate(*proto_args.calcargs,
                                      **proto_args.calckwargs))
            else:
                file_data[proto_args.accessor_string] = \
                    calculate(*proto_args.calcargs,
                              **proto_args.calckwargs)
            self_.changed_dirs.update(os.path.dirname(path) \
                                      for path in proto_args.paths)
            self_.update_caches()
            if hasattr(self_, 'import_'+data_name):
                return getattr(self_, 'import_'+data_name) \
                              (file_data[proto_args.accessor_string])
            return file_data[proto_args.accessor_string]
    get.__name__ = 'get_'+data_name
    get.__doc__  = 'Call calculate_'+data_name+ \
                   '() on a file path with caching magic.'
    return get

def _make_gather(data_name):
    '''Make a gather_ accessor for a FileDataBuffer that concurrently operates
    on a list of values for an argument,  instead of a single value.'''
    def gather(self_, *args, argi=0, **kwargs):
        # New keyword argument argi: which argument is to be made into a list.
        # If it is a number, then it corresponds to a positional argument (self
        # not included in the numbering). If it is a string, then it
        # corresponds to /any/ argument with the name of the string (which
        # could be a required positional argument). It may also be a list of
        # such indices, to listify for all of those.
        calculate = getattr(self_, 'calculate_'+data_name)
        # If cacheablep is False, then the results won't be permanently saved
        # in the data dict, and the import/export methods will be bypassed.
        cacheablep = not hasattr(calculate, 'non_cacheable')

        def unpack_args():
            '''A generator that will come up with a tuple of (args, ukwargs)
            until all the requested combinations have been done.
            '''
            # (also I have a strong feeling this could all be rewritten more
            #  concisely, but I have no idea how to make it happen)
            std_argi = copy.deepcopy(argi)
            if not hasattr(std_argi, '__iter__') or \
               isinstance(std_argi, str):
                std_argi = [std_argi]
            argspec = inspect.getfullargspec(calculate)
            lendefaults = len(argspec.defaults) \
                          if argspec.defaults is not None \
                          else 0
            # Using Functional CodeTM, do for all possible combinations of args
            # from the lists received for the listified args:
            for combo in itertools.product(
                *(((kwargs[argspec.args[i+1]] if i >= len(argspec.args) - \
                                                      lendefaults - 1 \
                    else args[i]) if isinstance(i, int) \
                   else kwargs[i] if isinstance(i, str) else None) \
                  for i in std_argi)):
                newargs   = list(copy.copy(args))
                newkwargs = copy.copy(kwargs)
                for i, value in zip(std_argi,
                                    (value for value in combo)):
                    if isinstance(i, int):
                        # -1 to account for self arg
                        if i >= len(argspec.args) - lendefaults - 1:
                            # again, +1 to account for self arg
                            newkwargs[argspec.args[i+1]] = value
                        else:
                            newargs[i] = value
                    elif isinstance(i, str):
                        newkwargs[i] = value
                yield (newargs, newkwargs)
        if MPISIZE == 1 or not self_.calculatingp:
            final_result = []
            for uargs, ukwargs in unpack_args():
                try:
                    final_result.append(getattr(self_, 'get_'+data_name) \
                                               (*uargs, **ukwargs))
                except KeyError:
                    pass
            return final_result
        MPITag = enum.Enum('MPITag', 'READY_TAG DONE_TAG WORK_TAG')
        class QUIT_MSG:
            pass
        # READY_TAG: used once by worker, to sync startup
        # DONE_TAG:  used by worker to pass result and request new job
        # WORK_TAG:  used by master to pass next path to worker
        # QUIT_MSG:  getting this instead of a path kills a worker thread
        # will be used to index into self_.data to retrieve final result:
        file_indices_list = []
        if MPIRANK == 0:
            ## helper functions
            def recv_result():
                return (MPICOMM.recv(source=MPI.ANY_SOURCE,
                                     tag=MPI.ANY_TAG,
                                     status=MPISTATUS),
                        MPISTATUS.Get_source(),
                        MPISTATUS.Get_tag())
            def save_result(index, proto_args, result_data, exportp=True):
                _DEBUG_OUT('about to save result for ' + \
                          proto_args.prime_path)
                proto_args.update_paths()
                file_data = self_.data.setdefault(proto_args.prime_hash,
                                                  {}) \
                                      .setdefault(data_name, {})
                if hasattr(self_, 'export_'+data_name) and \
                   exportp and cacheablep:
                    file_data[proto_args.accessor_string] = \
                        getattr(self_, 'export_'+data_name)(result_data)
                else:
                    file_data[proto_args.accessor_string] = result_data
                file_indices_list.append([index,
                                          proto_args.prime_hash,
                                          data_name,
                                          proto_args.accessor_string])
                _DEBUG_OUT('just saved result for ' + proto_args.prime_path)
            ## main loop
            for index, uargs_and_ukwargs in enumerate(unpack_args()):
                uargs, ukwargs = uargs_and_ukwargs
                proto_args = self_.proto_args(uargs, ukwargs)
                # lazy cache retrieval!
                if cacheablep:
                    for path in proto_args.paths:
                        self_.retrieve_data_from_cache(os.path.dirname(path))
                file_data = self_.data \
                                 .setdefault(proto_args.prime_hash, {}) \
                                 .setdefault(data_name, {})
                try:
                    assert cacheablep
                    save_result(index,
                                proto_args,
                                file_data[proto_args.accessor_string],
                                exportp=False)
                except (KeyError, AssertionError):
                    result, result_source, result_tag = recv_result()
                    _DEBUG_OUT('assigning ' + proto_args.prime_path + \
                              ' to ' + str(result_source))
                    MPICOMM.send([index, proto_args],
                                 dest=result_source, tag=MPITag.WORK_TAG)
                    _DEBUG_OUT('assignment sent')
                    if result_tag == MPITag.DONE_TAG:
                        save_result(*result)
                        self_.changed_dirs \
                             .update(os.path.dirname(path) \
                                     for path \
                                     in result[1].paths)
                        self_.update_caches()
            ## clean up workers once we run out of stuff to assign
            for _ in range(MPISIZE-1):
                result, result_source, result_tag = recv_result()
                _DEBUG_OUT('data received from '+str(result_source))
                if result_tag == MPITag.DONE_TAG:
                    save_result(*result)
                    self_.changed_dirs.update(os.path.dirname(path) \
                                              for path in result[1].paths)
                    self_.update_caches()
                _DEBUG_OUT('all files done. killing '+str(result_source))
                MPICOMM.send(QUIT_MSG,
                             dest=result_source, tag=MPITag.WORK_TAG)
                _DEBUG_OUT('worker killed')
        else:
            MPICOMM.send(None, dest=0, tag=MPITag.READY_TAG)
            while True:
                package = MPICOMM.recv(source=0, tag=MPITag.WORK_TAG)
                if package == QUIT_MSG:
                    break
                index, proto_args = package
                result_data = calculate(*proto_args.calcargs,
                                        **proto_args.calckwargs)
                MPICOMM.send(copy.copy([index,
                                        proto_args,
                                        result_data]),
                             dest=0, tag=MPITag.DONE_TAG)
        # Synchronize everything that could have possibly changed:
        self_.data = MPICOMM.bcast(self_.data, root=0)
        self_.file_paths = MPICOMM.bcast(self_.file_paths, root=0)
        self_.cache_paths = MPICOMM.bcast(self_.cache_paths, root=0)
        file_indices_list = MPICOMM.bcast(file_indices_list, root=0)
        # All threads should be on the same page at this point.
        file_indices_list.sort(key=lambda x: x[0])
        # Sort by the index produced by enumerating unpack_args() above.
        if hasattr(self_, 'import_'+data_name) and cacheablep:
            retval = [getattr(self_, 'import_'+data_name) \
                             (self_.data[indices[1]] \
                                        [indices[2]] \
                                        [indices[3]]) \
                      for indices in file_indices_list]
        else:
            retval = [self_.data[indices[1]][indices[2]][indices[3]] \
                      for indices in file_indices_list]
        if not cacheablep:
            for indices in file_indices_list:
                del self_.data[indices[1]][indices[2]][indices[3]]
        return retval
    # Why doesn't this *work*?
    gather.__name__ = 'gather_'+data_name
    gather.__doc__  = 'Call calculate_'+data_name+ \
                      '() on a list of file paths with concurrency magic.'
    return gather


class FileDataBuffer():
    '''Singleton class that stores information about files, to be used as a data
    buffer. Its purpose is to intelligently abstract the retrieval of
    information about files in such a way that the information is cached and/or
    buffered in the process. The only methods in it that should ideally be used
    externally are the data retrieval methods, and out of those ideally just the
    caching ones (currently get_ and gather_ methods). In practice,
    update_caches() is also used externally to force a cache update.

    Internally, it holds a data dict, which indexes all the data first by the
    MD5 hash of the contents of the file that produced the data, then the type
    of data it is (e.g. RMSD or score), and finally the parameters that
    produced the data (like the particular weights on the scorefunction for a
    score). It also holds a dict of paths to files, the contents hashes of these
    files, and the mtime of the file at which the contents hash was produced, so
    that if it is asked about a file whose hash is already known, it does not
    need to recalculate it. Finally, it also holds a dict mapping the caches
    it's already loaded to their mtimes at which they were loaded, so that it
    does not load a cache unless it's changed.

    On disk, the data is cached in a .zaluski_cache file that contains a JSON
    array of the data dict and the file paths dict, each with only the entries
    for the file files in the same folder as the cache. If caching is turned on,
    whenever the buffer generates data for a file, it will check whether the
    file's folder has a cache with that data yet, and if not, it will write one
    (making sure to load any cached data in that folder that already exists).
    Caching is done in a thread-safe manner, with file locks. Throughout the
    reading, updating, and writing of a cache, the buffer will maintain an
    exclusive lock on the cache, so that it doesn't get updated on disk in
    between its reading and updating/writing, which would cause the first
    update to get lost.

    The ability to get a particular type of data is added to the buffer by
    defining a calculate_ method. Corresponding get_ and gather_ methods are
    created dynamically at initialization, which add caching to the calculate_
    method, and in the case of gather_, concurrently operate on lists given for
    arguments that normally don't take them. Because PDB hashes are used as the
    top-level keys in the data dict, each calculate_ method should take at
    least one PDB, either as a stream or a path. See the comment above the
    calculate_ methods for more details.

    The internal settings calculatingp and plottingp work like this:

      - calculatingp = False: The buffer will only retrieve cached information,
          never calculating its own data or saving caches. get_ operations will
          return a KeyError if the data for a file is not in the caches it's
          loaded, and gather_ operations will leave a file's data out of the
          list they return if they can't find it.
      - calculatingp = True, cachingp = False: The buffer will calculate new
          information if it doesn't have it, but never save it. Useful if
          real-time caching is taking too much time, and if instead you want to
          do it manually by calling update_caches(force=True) at select times.
          Note that with cachingp = False, update_caches() doesn't do anything
          unless you call it with force=True. (This is the case with
          calculatingp = False as well, making it the only real effect of
          cachingp in that case.)
      - calculatingp = True, cachingp = True: The buffer will calculate new
          information if it doesn't have it, and save it to disk immediately
          all of the time.

    The structure of the buffer's data variables, although summarized above,
    can be stated more succinctly as:

    self.data:
    { content_key : { data_function_name :
                        { data_function_params_tuple : data } } }
    self.file_paths:
    { file_path : { 'mtime' : mtime_at_last_hashing,
                    'hash'  : contents_hash } }
    self.cache_paths:
    { cache_file_dir_path : mtime_at_last_access }
    '''

    ## Core functionality
    def __init__(self, *, filename='.zaluski_cache', version=0):
        self.filename = filename
        self.version = version
        self.calculatingp = True
        self.cachingp = MPIRANK == 0
        self.data = {}
        self.file_paths = {}
        self.cache_paths = {}
        self.changed_dirs = set()
        # Monkey patch in the data accessors:
        attribute_names = dir(self)
        for data_name in (attribute_name[10:] \
                          for attribute_name in attribute_names \
                          if attribute_name.startswith('calculate_')):
            calcfxn = getattr(self, 'calculate_'+data_name)

            # Rationale for this warning: The point at which the buffer
            # actually fails when you forget a self arg is deep inside
            # _ProtoArgs, which is very cryptic and hard to debug.
            if inspect.getfullargspec(calcfxn).args[0] != 'self':
                warnings.warn('Added method calculate_' + data_name + \
                              ' has no self arg.', SyntaxWarning)

            # specialized import/export functions
            special_porters = \
                (('returns_ndarray',    (decompress_ndarray,
                                         compress_ndarray)),
                 ('returns_pickleable', (decompress_pickleable,
                                         compress_pickleable)))
            for attribute, (import_func, export_func) in special_porters:
                if hasattr(calcfxn, attribute):
                    setattr(self, 'import_'+data_name,
                            types.MethodType(lambda self_, x, __f=import_func:
                                                 __f(x),
                                             self))
                    setattr(self, 'export_'+data_name,
                            types.MethodType(lambda self_, x, __f=export_func:
                                                 __f(x),
                                             self))
                    break
            # get_ function
            setattr(self, 'get_'+data_name,
                    types.MethodType(_make_get(data_name),
                                     self))
            # gather_ function
            setattr(self, 'gather_'+data_name,
                    types.MethodType(_make_gather(data_name),
                                     self))
    def retrieve_data_from_cache(self, dirpath, _update_caches_fd=None):
        '''Retrieves data from a cache file of a directory. The data in the file
        is a JSON of a data dict and a file_paths list of a FileDataBuffer,
        except instead of file paths, it has just filenames.'''
        if MPIRANK != 0:
            return
        absdirpath = os.path.abspath(dirpath)
        try:
            cache_path = os.path.join(absdirpath, self.filename)
            diskmtime = os.path.getmtime(cache_path)
            ourmtime = self.cache_paths.setdefault(absdirpath, 0)
            if diskmtime > ourmtime:
                file_contents = None
                retrieved = None
                if _update_caches_fd is not None:
                    pos = _update_caches_fd.tell()
                    _update_caches_fd.seek(0)
                    file_contents = _update_caches_fd.read()
                    if file_contents:
                        _DEBUG_OUT('cache at '+_update_caches_fd.name+' read')
                    _update_caches_fd.seek(pos)
                else:
                    with open(cache_path, 'r') as cache_file:
                        file_contents = cache_file.read()
                        _DEBUG_OUT('cache at '+cache_file.name+' read')
                try:
                    retrieved = json.loads(file_contents)
                except json.JSONDecodeError:
                    if file_contents:
                        print('Could not interpret cache file at '+cache_path)
                        print('You should probably delete it.')
                    return
                if retrieved is not None:
                    _DEBUG_OUT('cache retrieved')
                    diskdata, disk_file_paths = retrieved
                    for content_key, content in diskdata.items():
                        for data_name, data_keys in content.items():
                            for data_key, data in data_keys.items():
                                self.data.setdefault(content_key, {})
                                self.data[content_key].setdefault(data_name, {})
                                self.data[content_key] \
                                         [data_name] \
                                         [data_key] = data
                    for name, info_pair in disk_file_paths.items():
                        _DEBUG_OUT('saving info for '+name)
                        path = os.path.join(absdirpath, name)
                        our_info_pair = self.file_paths.get(path, None)
                        ourfilemtime = None
                        if our_info_pair is not None:
                            ourfilemtime = our_info_pair['mtime']
                        else:
                            ourfilemtime = 0
                        if info_pair['mtime'] > ourfilemtime:
                            self.file_paths[path] = info_pair
            self.cache_paths[absdirpath] = diskmtime
        except FileNotFoundError:
            pass
    def update_caches(self, force=False):
        '''Updates the caches for every directory the cache knows about. This
        method is thread-safe.'''
        if (not (self.cachingp or force)) or MPIRANK != 0:
            return
        dir_paths = {}
        for path in self.file_paths:
            dir_path, name = os.path.split(path)
            if dir_path in self.changed_dirs:
                dir_paths.setdefault(dir_path, set())
                dir_paths[dir_path].add(name)
        for dir_path, names in dir_paths.items():
            cache_path = os.path.join(dir_path, self.filename)
            cache_file = None
            try:
                cache_file = open(cache_path, 'r+')
            except FileNotFoundError:
                cache_file = open(cache_path, 'w+')
            while True:
                try:
                    fcntl.flock(cache_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.05)
            self.retrieve_data_from_cache(dir_path,
                                          _update_caches_fd=cache_file)
            dir_data = {}
            dir_file_paths = {}
            for name in names:
                path = os.path.join(dir_path, name)
                dir_file_paths[name] = self.file_paths[path]
                ourhash = dir_file_paths[name]['hash']
                try:
                    dir_data[ourhash] = self.data[ourhash]
                except KeyError:
                    pass
                #_DEBUG_OUT('  updated cache with data for file at ', path)
            cache_file.write(json.dumps([dir_data, dir_file_paths], indent=4))
            fcntl.flock(cache_file, fcntl.LOCK_UN)
            cache_file.close()
            _DEBUG_OUT('wrote cache file at ', cache_path)
            self.cache_paths[dir_path] = os.path.getmtime(cache_path)
            self.changed_dirs = set()

    ## Constructors for auxiliary classes
    def _proto_args(self, calcfxn, args, kwargs):
        '''Given a set of args for a get_ accessor method, generate an object
        that FileDataBuffer accessor methods can use to both call calculate_
        methods and index into self.data and self.paths, based on
        EncapsulatedFile encapsulations of all of the paths passed to it. It
        stores a list and dict of the arg values with paths replaced by
        EncapsulatedFiles, a corresponding list and dict of arg types ('path',
        'stream', or 'misc' depending on what arg is requested by the
        calculate_ method), and the path to the first file in the args, its
        contents hash, and the accessor string for the args (for indexing into
        self.data). It can also be iterated over to provide the arg values in
        the sequence in which they were originally part of the calculate_
        method, but this feature is not currently in use.'''
        return _ProtoArgs(calcfxn,
                          self._encapsulate_file,
                          args, kwargs)

    def _encapsulate_file(self, path):
        '''Returns an object that in theory contains the absolute path, mtime,
        contents hash, and maybe contents stream of a file. The first three are
        accessed directly, while the last is accessed via an accessor method,
        so that it can be retrieved if necessary. Creating and updating the
        object both update the external FileDataBuffer's info on that pdb.'''
        return EncapsulatedFile(path,
                                self.file_paths
                                    .setdefault(os.path.abspath(path), {}))

zaluski
=======

zaluski provides a buffer class, ``FileDataBuffer``, that is designed to be
augmented with methods that compute information about files. The buffer will
then intelligently derive new versions of these methods that cache their
results, and/or operate concurrently on multiple sets of files or parameters.

It is named after the Załuski brothers, two bishops who founded one of Europe's
first public libraries.

Installation
------------

In the same directory as ``setup.py``, run ::

    $ pip install . --user

or ::

    $ sudo pip install .

Basic usage and features
------------------------

Create a subclass of ``zaluski.FileDataBuffer``, and add a method to it that
begins with ``calculate_``, and takes an argument called ``path``::

    import zaluski

    class MyDataBuffer(zaluski.FileDataBuffer):
        def calculate_file_size(self, path):
            return len(open(path).read())

The new subclass will then automatically create an analogous method beginning
with ``get_``, which performs the same function as the ``calculate_`` method,
but also caches the result (in a hidden file). Imagine that in our directory we
have the following file, ``example.txt``::

    This text contains forty-three characters.

Then, using the MyDataBuffer we created earlier, we have:

>>> buf = MyDataBuffer()
>>> buf.get_file_size('example.txt')
43

This result is stored in a cache file ``.zaluski_cache`` in the same directory as
``example.txt``, and all future instances of ``MyDataBuffer`` will remember it
(unless you change the buffer's version number, or delete the cache). If the
file ever changes, the buffer will automatically recompute its results.

More features
-------------

* The buffer auto-detects which arguments relate to files based on the argument
  names, allowing for paths, streams, and ordered or unordered collections of
  either. Future versions will expect you to indicate this with annotations,
  rather than using specific argument names.
* The buffer will also generate ``gather_`` versions of your methods, which can
  run the method combinatorially on a set of arguments, and, if possible,
  distribute the work over multiple processors.
* You can define ``import_`` and ``export_`` methods that determine how the
  result of a method is serialized.
* Both the computing and caching features of the buffer can be disabled at will
  on a method-by-method basis.

To do
-----

* Redo type hints as parameter annotations, rather than function names. Maybe
  also redo the import/export decorators.
* Give more fine-grained control over frequency of caching.
* Improve concurrency for small amounts of processors (especially two).

License
-------

This project is licensed under GPLv3.

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import hashlib
import warnings
import six
import uuid


_ALPHABET_27 = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


_HASH_LEN2 = 32

if six.PY3:
    _stringlike = (str, bytes)  # NOQA
else:
    _stringlike = (basestring, bytes)  # NOQA

if six.PY3:
    def _int_to_bytes(int_):
        length = max(4, int_.bit_length())
        bytes_ = int_.to_bytes(length, byteorder='big')
        # bytes_ = int_.to_bytes(4, byteorder='big')
        # int_.to_bytes(8, byteorder='big')  # TODO: uncomment
        return bytes_

    def _bytes_to_int(bytes_):
        int_ = int.from_bytes(bytes_, 'big')
        return int_
else:
    def _py2_to_bytes(int_, length, byteorder='big'):
        h = '%x' % int_
        s = ('0' * (len(h) % 2) + h).zfill(length * 2).decode('hex')
        bytes_ =  s if byteorder == 'big' else s[::-1]
        return bytes_

    import codecs
    def _int_to_bytes(int_):
        length = max(4, int_.bit_length())
        bytes_ = _py2_to_bytes(int_, length, 'big')
        # bytes_ = struct.pack('>i', int_)
        return bytes_

    def _bytes_to_int(bytes_):
        int_ = int(codecs.encode(bytes_, 'hex'), 16)
        # int_ = struct.unpack('>i', bytes_)[0]
        # int_ = struct.unpack_from('>L', bytes_)[0]
        return int_


def hash_data(data, hashlen=None, alphabet=None):
    r"""
    Get a unique hash depending on the state of the data.

    Args:
        data (object): any sort of loosely organized data
        hashlen (None): (default = None)
        alphabet (None): (default = None)

    Returns:
        str: text -  hash string

    Example:
        >>> counter = [0]
        >>> failed = []
        >>> def check_hash(input_, want=None):
        ...     count = counter[0] = counter[0] + 1
        ...     got = hash_data(input_)
        ...     #print('({}) {}'.format(count, got))
        ...     if want is not None and not got.startswith(want):
        ...         failed.append((got, input_, count, want))
        >>> check_hash('1', 'wuvrng')
        >>> check_hash(['1'], 'dekbfpby')
        >>> check_hash(tuple(['1']), 'dekbfpby')
        >>> check_hash(b'12', 'marreflbv')
        >>> check_hash([b'1', b'2'], 'nwfs')
        >>> check_hash(['1', '2', '3'], 'arfrp')
        >>> #check_hash(['1', np.array([1,2,3]), '3'], 'uyqwcq')
        >>> check_hash('123', 'ehkgxk')
        >>> check_hash(zip([1, 2, 3], [4, 5, 6]), 'mjcpwa')
    """
    if alphabet is None:
        alphabet = _ALPHABET_27
    if hashlen is None:
        hashlen = _HASH_LEN2
    hasher = hashlib.sha512()
    _update_hasher(hasher, data)
    # Get a 128 character hex string
    text = hasher.hexdigest()
    if alphabet == 'hex':
        hashstr2 = text
    else:
        # Shorten length of string (by increasing base)
        hashstr2 = _convert_hexstr_to_bigbase(text, alphabet)
    # Truncate
    text = hashstr2[:hashlen]
    return text


def _update_hasher(hasher, data):
    """
    This is the clear winner over the generate version.
    Used by hash_data

    Example:
        >>> hasher = hashlib.sha256()
        >>> data = [1, 2, ['a', 2, 'c']]
        >>> _update_hasher(hasher, data)
        >>> print(hasher.hexdigest())
        31991add5389e4bbca49530dfaee96f31035e3df4cd4fd4121a186728532c5b8

    """
    if isinstance(data, (tuple, list, zip)):
        needs_iteration = True
    # elif (util_type.HAVE_NUMPY and isinstance(data, np.ndarray) and
    #       data.dtype.kind == 'O'):
    #     # ndarrays of objects cannot be hashed directly.
    #     needs_iteration = True
    else:
        needs_iteration = False

    if needs_iteration:
        # try to nest quickly without recursive calls
        SEP = b'SEP'
        iter_prefix = b'ITER'
        iter_ = iter(data)
        hasher.update(iter_prefix)
        try:
            for item in iter_:
                prefix, hashable = _covert_to_hashable(data)
                binary_data = SEP + prefix + hashable
                hasher.update(binary_data)
        except TypeError:
            # need to use recursive calls
            # Update based on current item
            _update_hasher(hasher, item)
            for item in iter_:
                # Ensure the items have a spacer between them
                hasher.update(SEP)
                _update_hasher(hasher, item)
    else:
        prefix, hashable = _covert_to_hashable(data)
        binary_data = prefix + hashable
        hasher.update(binary_data)


class HashableExtensions():
    """
    Helper class for managing non-builtin (e.g. numpy) hash types
    """
    def __init__(self):
        self.extensions = []

    def register(self, hash_type):
        def _wrap(hash_func):
            self.extensions.append((hash_type, hash_func))
            return hash_func
        return _wrap

    def lookup(self, data):
        for hash_type, hash_func in self.extensions:
            if isinstance(data, hash_type):
                return hash_func

    def _register_numpy_extensions(self):
        """
        Numpy extensions are builtin
        """
        @self.register(np.ndarray)
        def hash_numpy_array(data):
            if data.dtype.kind == 'O':
                msg = 'hashing ndarrays with dtype=object is unstable'
                warnings.warn(msg, RuntimeWarning)
                hashable = data.dumps()
            else:
                hashable = data.tobytes()
            prefix = b'NDARR'
            return hashable, prefix

        @self.register((np.int64, np.int32))
        def _hash_numpy_int(data):
            return _covert_to_hashable(int(data))

        @self.register((np.float64, np.float32))
        def _hash_numpy_float(data):
            a, b = float(data).as_integer_ratio()
            hashable = (a.to_bytes(8, byteorder='big') +
                        b.to_bytes(8, byteorder='big'))
            prefix = b'FLT'
            return hashable, prefix

_HASHABLE_EXTENSIONS = HashableExtensions()


try:
    import numpy as np
    _HASHABLE_EXTENSIONS._register_numpy_extensions()
except ImportError:
    pass


def _covert_to_hashable(data):
    r"""
    Args:
        data (object): arbitrary data

    Returns:
        tuple(bytes, bytes): prefix, hashable:
            indicates the


    Example:
        >>> import ubelt as ub
        >>> assert _covert_to_hashable('string') == (b'', b'string')
        >>> assert _covert_to_hashable(1) == (b'', b'\x00\x00\x00\x01')
        >>> assert _covert_to_hashable(1.0) == (b'', b'1.0')
    """
    if data is None:
        hashable = b''
        prefix = b'NONE'
    elif isinstance(data, six.binary_type):
        hashable = data
        prefix = b'TXT'
    elif isinstance(data, six.text_type):
        # convert unicode into bytes
        hashable = data.encode('utf-8')
        prefix = b'TXT'
    elif isinstance(data, uuid.UUID):
        hashable = data.bytes
        prefix = b'UUID'
    elif isinstance(data, int):
        # warnings.warn('Hashing ints is slow, numpy is prefered')
        hashable = _int_to_bytes(data)
        # hashable = data.to_bytes(8, byteorder='big')
        prefix = b'INT'
    elif isinstance(data, float):
        hashable = repr(data).encode('utf8')
        prefix = b'FLT'
    else:
        hash_func = _HASHABLE_EXTENSIONS.lookup(data)
        if hash_func is not None:
            hashable, prefix = hash_func(data)
        else:
            raise TypeError('unknown hashable type=%r' % (type(data)))
    return prefix, hashable


def _convert_hexstr_to_bigbase(hexstr, alphabet):
    r"""
    Packs a long hexstr into a shorter length string with a larger base

    Example:
        >>> newbase_str = _convert_hexstr_to_bigbase(
        ...     'ffffffff', _ALPHABET_27, len(_ALPHABET_27))
        >>> print(newbase_str)
        vxlrmxn

    Sympy:
        >>> import sympy as sy
        >>> # Determine the length savings with lossless conversion
        >>> consts = dict(hexbase=16, hexlen=256, bigbase=27)
        >>> symbols = sy.symbols('hexbase, hexlen, bigbase, newlen')
        >>> haexbase, hexlen, bigbase, newlen = symbols
        >>> eqn = sy.Eq(16 ** hexlen,  bigbase ** newlen)
        >>> newlen_ans = sy.solve(eqn, newlen)[0].subs(consts).evalf()
        >>> print('newlen_ans = %r' % (newlen_ans,))
        >>> # for a 27 char alphabet we can get 216
        >>> print('Required length for lossless conversion len2 = %r' % (len2,))
        >>> def info(base, len):
        ...     bits = base ** len
        ...     print('base = %r' % (base,))
        ...     print('len = %r' % (len,))
        ...     print('bits = %r' % (bits,))
        >>> info(16, 256)
        >>> info(27, 16)
        >>> info(27, 64)
        >>> info(27, 216)
    """
    bigbase = len(alphabet)
    x = int(hexstr, 16)  # first convert to base 16
    if x == 0:
        return '0'
    sign = 1 if x > 0 else -1
    x *= sign
    digits = []
    while x:
        digits.append(alphabet[x % bigbase])
        x //= bigbase
    if sign < 0:
        digits.append('-')
        digits.reverse()
    newbase_str = ''.join(digits)
    return newbase_str


def hash_file(fpath, blocksize=65536, hasher=None, stride=1):
    r"""
    For better hashes use hasher=hashlib.sha256, and keep stride=1

    Args:
        fpath (str):  file path string
        blocksize (int): 2 ** 16. Affects speed of reading file
        hasher (None):  defaults to sha1 for fast (but non-robust) hashing
        stride (int): strides > 1 skip data to hash, useful for faster
                      hashing, but less accurate, also makes hash dependant on
                      blocksize.

    References:
        http://stackoverflow.com/questions/3431825/md5-checksum-of-a-file
        http://stackoverflow.com/questions/5001893/when-to-use-sha-1-vs-sha-2

    Example:
        >>> from clab.util import *
        >>> import ubelt as ub
        >>> from os.path import join
        >>> fpath = join(ub.ensure_app_cache_dir('ubelt'), 'tmp.txt')
        >>> ut.write_to(fpath, 'foobar')
        >>> hash_file(fpath)
        oscjatwzheaqcvlpdowhzhowwikupotgwgqyrpnnvewhvxkxxjtivld
    """
    if hasher is None:
        hasher = hashlib.sha256()
    with open(fpath, 'rb') as file:
        buf = file.read(blocksize)
        if stride > 1:
            # skip blocks when stride is greater than 1
            while len(buf) > 0:
                hasher.update(buf)
                file.seek(blocksize * (stride - 1), 1)
                buf = file.read(blocksize)
        else:
            # otherwise hash the entire file
            while len(buf) > 0:
                hasher.update(buf)
                buf = file.read(blocksize)
        hexid = hasher.hexdigest()
        hashid = _convert_hexstr_to_bigbase(hexid, _ALPHABET_27)
        return hashid

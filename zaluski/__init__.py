# zaluski/__init__.py
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

from .filedatabuffer import (
    returns_ndarray,
    returns_pickleable,
    non_cacheable,
    FileDataBuffer
)

__all__ = [
    'returns_ndarray',
    'returns_pickleable',
    'non_cacheable',
    'FileDataBuffer'
]

__title__ = 'zaluski'
__author__ = 'Maria Szegedy'
__license__ = 'GPLv3'
__copyright__ = '2019, Maria Szegedy'

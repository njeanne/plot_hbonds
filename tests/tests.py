#!/usr/bin/env python3

__author__ = 'Nicolas Jeanne'
__license__ = 'GNU General Public License'
__version__ = '1.0.0'
__email__ = 'jeanne.n@chu-toulouse.fr'

import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()

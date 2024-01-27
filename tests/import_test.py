import unittest

class TestImport(unittest.TestCase):

    def test_imports(self):
        try:
            import cbench
            import cbench.benchmark
            import cbench.data
            import cbench.codecs
            # pybind11 libs
            import cbench.rans
            import cbench.ans
            import cbench.ar
        except:
            self.fail()


if __name__ == '__main__':
    unittest.main()
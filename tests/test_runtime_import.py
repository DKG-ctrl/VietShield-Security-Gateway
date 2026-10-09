import unittest


class RuntimeImportTests(unittest.TestCase):
    def test_runtime_can_be_imported_as_package(self):
        from scripts.runtime import load_frozen_gateway

        self.assertTrue(callable(load_frozen_gateway))


if __name__ == "__main__":
    unittest.main()

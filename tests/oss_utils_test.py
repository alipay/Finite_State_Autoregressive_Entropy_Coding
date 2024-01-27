import unittest

import os
import shutil
from configs.oss_utils import OSSUtils

class TestOSSUtils(unittest.TestCase):

    def test_file(self):
        oss = OSSUtils()
        test_upload_file = "README.md"
        test_download_file = "test_file"
        oss.upload(test_upload_file, test_upload_file)
        oss.download(test_upload_file, test_download_file)

        with open(test_upload_file, 'r') as f:
            up_content = f.read()

        with open(test_download_file, 'r') as f:
            dn_content = f.read()
        
        os.remove(test_download_file)

        self.assertSequenceEqual(up_content, dn_content)

    def test_dir(self):
        oss = OSSUtils()
        test_upload_dir = "tools"
        test_download_dir = "test_dir"
        oss.upload_directory(test_upload_dir, test_upload_dir)
        oss.download_directory(test_upload_dir, test_download_dir)

        # TODO: check content
        # self.assertSequenceEqual(up_content, dn_content)
        shutil.rmtree(test_download_dir)

        test_upload_dir = "tools"
        test_download_dir = "test_dir"
        oss.sync_directory(test_upload_dir, test_upload_dir)
        oss.sync_directory(test_upload_dir, test_download_dir)

        # TODO: check content
        # self.assertSequenceEqual(up_content, dn_content)
        shutil.rmtree(test_download_dir)

if __name__ == '__main__':
    unittest.main()
import unittest
import random
import numpy as np
import time

from cbench.ar import ar_3way_mean, ar_3way_mean_array, ar_linear_op, ar_lookup_op
from cbench.ar import autoregressive_transform, autoregressive_transform_3way, autoregressive_transform_3way_static, autoregressive_transform_3way_staticfunc, autoregressive_transform_3way_tpl, autoregressive_transform_3way_op_tpl

class TestAutoregressive(unittest.TestCase):

    def test_ar_transform(self):
        data = np.random.randint(0, 256, (4, 3, 32, 32))
        ar_offsets = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        ar_func = lambda i1, i2, i3 : (i1 + i2 + i3) / 3
        ar_func_tpl = lambda input : (input[0] + input[1] + input[2]) / 3

        def autoregressive_transform_3way_np(data):
            output = np.zeros_like(data)
            output[1:, :, :] = data[1:, :, :] - data[:-1, :, :]
            output[:, 1:, :] = data[:, 1:, :] - data[:, -1:, :]
            output[:, :, 1:] = data[:, :, 1:] - data[:, :, -1:]
            return output

        time_start = time.time()
        transformed_np = autoregressive_transform_3way_np(data)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way_np: {time.time()-time_start}")

        time_start = time.time()
        transformed = autoregressive_transform_3way_static(data, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way_static: {time.time()-time_start}")
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())
        # self.assertTrue((transformed==transformed_np).all())

        time_start = time.time()
        transformed = autoregressive_transform_3way_staticfunc(data, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way_staticfunc: {time.time()-time_start}")
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())

        time_start = time.time()
        transformed = autoregressive_transform_3way(data, ar_3way_mean, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way(cfunc): {time.time()-time_start}")
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())

        time_start = time.time()
        transformed = autoregressive_transform_3way(data, ar_func, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way: {time.time()-time_start}")
        # original_data = autoregressive_transform_3way(transformed, ar_func, ar_offsets)
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())

        time_start = time.time()
        transformed = autoregressive_transform_3way_tpl(data, ar_func_tpl, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way_tpl: {time.time()-time_start}")
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())

        time_start = time.time()
        transformed = autoregressive_transform_3way_tpl(data, ar_3way_mean_array, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way_tpl(cfunc_array): {time.time()-time_start}")
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())

        ar_func_dyn = ar_linear_op([1./3]*3, 0.)
        # print(ar_func_dyn([1.,2.,3.]))

        time_start = time.time()
        transformed = autoregressive_transform_3way_tpl(data, ar_func_dyn, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way_tpl(cfunc): {time.time()-time_start}")
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())

        time_start = time.time()
        transformed = autoregressive_transform_3way_op_tpl(data, ar_func_dyn, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform_3way_op_tpl: {time.time()-time_start}")
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())

        time_start = time.time()
        transformed = autoregressive_transform(data, ar_func_dyn, ar_offsets)
        # print(data, transformed)
        print(f"Time of autoregressive_transform: {time.time()-time_start}")
        # self.assertSequenceEqual(transformed.tolist(), transformed_np.tolist())

        # self.assertSequenceEqual(data.tolist(), original_data.tolist())

        
if __name__ == '__main__':
    unittest.main()
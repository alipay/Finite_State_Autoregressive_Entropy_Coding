import math
import unittest

from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot, NamedParam

class BasicTestClass(object):
    def __init__(self, a, *args, b=1, pow=None, other_obj=None, **kwargs):
        self.a = a # an arg
        self.b = b # a normal kwarg
        self.pow = pow # optional kwarg
        self.other_obj = other_obj # class kwarg
    
    def foo(self, c):
        if self.other_obj is not None:
            if isinstance(self.other_obj, (list, tuple)):
                for obj in self.other_obj:
                    c = obj.foo(c)
            elif isinstance(self.other_obj, dict):
                for obj in self.other_obj.values():
                    c = obj.foo(c)
            else:
                c = self.other_obj.foo(c)
        val = (self.a + c) / self.b
        if self.pow is not None:
            val = math.pow(val, self.pow)
        return val

class TestClassBuilder(unittest.TestCase):

    def test_basic(self):
        # test build class
        cb = ClassBuilder(BasicTestClass, 4, b=2, any_list=[1, 2, 3], any_dict=dict(a=2, b=3)) #, any_class=Test(0))
        self.assertEqual(cb.build_class().foo(6), 5)

        # test serialize and deserialize (not working for now!)
        # test_class_string = cb.to_string()
        # print(test_class_string)
        # cb = eval(test_class_string)
        # self.assertEqual(cb.build_class().foo(6), 5)

        # test update args
        cb.update_args(2, b=4)
        self.assertEqual(cb.build_class().foo(6), 2)
        cb.update_args(2, 4)
        self.assertEqual(cb.build_class().foo(6), 2)
        cb.update_args(4, b=2)
        self.assertEqual(cb.build_class().foo(6), 5)

        # test nested cb
        nested_cb = ClassBuilder(BasicTestClass, 5, b=5, other_obj=cb)
        print(nested_cb)
        self.assertEqual(nested_cb.build_class().foo(6), 2)
        nested_cb.update_args(7, b=4)
        self.assertEqual(nested_cb.build_class().foo(6), 3)
        nested_cb.update_args(7, 3)
        self.assertEqual(nested_cb.build_class().foo(6), 4)

        # nested serialize and deserialize (not working for now!)
        # nested_class_string = nested_cb.to_string()
        # print(nested_class_string)
        # nested_cb = eval(nested_class_string)
        # self.assertEqual(nested_cb.build_class().foo(6), 2)

    def test_slots(self):
        # slot on args
        cb = ClassBuilder(BasicTestClass, 4, 
            b=ParamSlot("b", default=2, choices=list(range(1, 10)))
        ) #, any_class=Test(0))
        self.assertEqual(cb.build_class().foo(6), 5)
        cb_batch = cb.batch_update_slot_params(b=cb.SLOT_ALL_CHOICES)
        for cb_single, b in zip(cb_batch, range(1, 10)):
            self.assertEqual(cb_single.build_class().foo(6), 10/b)

        # param group slots
        cb = ClassBuilder(BasicTestClass, 4, 
            b=2,
            param_group_slots=[
                ParamSlot("group_a", default=dict(a=4), choices=[NamedParam('{}'.format(a), dict(a=a)) for a in range(10)]),
                ParamSlot("group_b", default=dict(b=2), choices=[NamedParam('{}'.format(b), dict(b=b)) for b in range(1,10)]),
            ],
        ) #, any_class=Test(0))
        # self.assertEqual(cb.build_class().foo(6), 5)
        cb_batch = cb.batch_update_slot_params(group_a=cb.SLOT_ALL_CHOICES, group_b=cb.SLOT_ALL_CHOICES)
        
    # def test_names(self):


if __name__ == '__main__':
    unittest.main()
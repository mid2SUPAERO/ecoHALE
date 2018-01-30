import importlib
import pkgutil
import unittest
from openaerostruct.tests.utils import get_default_lifting_surfaces, run_test


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        try:
            results[full_name] = importlib.import_module(full_name)
            if recursive and is_pkg:
                results.update(import_submodules(full_name))
        except:
            print(full_name, 'not found')
    return results

import openaerostruct
results = import_submodules(openaerostruct)

from six import iteritems

caps_names = ['fea', 'vlm', 'as', 'rhs', 'cp', 'ks']
skip_comps = ['fea_length_comp', 'fea_vonmises_comp']

lifting_surfaces = get_default_lifting_surfaces()

class Test(unittest.TestCase):

    def test(self):
        for key, data in iteritems(results):
            if '_comp' in key:
                file_comp_name = key.split('.')[-1]
                split_names = file_comp_name.split('_')
                capped_names = []
                for name in split_names:
                    if name in caps_names:
                        capped_name = name.upper()
                    else:
                        capped_name = name.capitalize()
                    capped_names.append(capped_name)
                comp_name = ''.join(capped_names)
                print(comp_name)

                print(file_comp_name)
                if 'vlm_eval' in file_comp_name:
                    exec_string = 'comp = ' + key + '.' + comp_name + '(num_nodes=1, lifting_surfaces=lifting_surfaces, eval_name="coll_pts", num_eval_points=5)'
                elif 'vlm_freestream_vel_comp' in file_comp_name:
                    exec_string = 'comp = ' + key + '.' + comp_name + '(num_nodes=1, size=5)'
                elif 'array_expansion' in file_comp_name:
                    exec_string = 'comp = ' + key + '.' + comp_name + '(shape=(8, 2), expand_indices=[1], in_name="in", out_name="out")'
                elif 'bspline_comp' in file_comp_name:
                    exec_string = 'comp = ' + key + '.' + comp_name + '(num_nodes=4, num_control_points=4, num_points=10, bspline_order=3, in_name="in", out_name="out", distribution="sine")'
                elif 'division_comp' in file_comp_name or 'product_comp' in file_comp_name:
                    exec_string = 'comp = ' + key + '.' + comp_name + '(shape=(16,), in_name1="in", in_name2="in2", out_name="out")'
                elif 'linear_comb_comp' in file_comp_name:
                    exec_string = 'comp = ' + key + '.' + comp_name + '(shape=(16, 4, 3), in_names=["in"], out_name="out")'
                elif 'scalar_expansion_comp' in file_comp_name:
                    exec_string = 'comp = ' + key + '.' + comp_name + '(shape=(16, 4, 3), in_name="in", out_name="out")'
                else:
                    exec_string = 'comp = ' + key + '.' + comp_name + '(num_nodes=1, lifting_surfaces=lifting_surfaces)'
                exec(exec_string, globals())

                if file_comp_name not in skip_comps:
                    run_test(self, comp)

if __name__ == '__main__':
    unittest.main()

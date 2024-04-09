import os
import sys
from setuptools import setup, find_packages
import subprocess


package_basename = 'acm'
package_dir = os.path.join(os.path.dirname(__file__), package_basename)
sys.path.insert(0, package_dir)

# compile C code under src/
subprocess.call(['make'])

# install package
setup(
    name=package_basename,
    packages=find_packages(),
    include_package_data=True,
)
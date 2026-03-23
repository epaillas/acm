# The unit test 

This folder contains the unit tests for the `acm` package. 
They are designed to check the validity of the functions and classes of the package.

However, given the nature of the package and the data it uses, some of those tests are not automated.

If a script name starts with `test_`, it is a test script that will be run by `pytest` and will be automatically checked for errors.
If a script name does not start with `test_`, it is a script that is meant to be run manually, and the results are to be checked by the user every time a change is made to the package.
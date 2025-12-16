# Code quality tools for developper

## Install quality tools


```console
source acm_quality_init.sh
```
 
Local user installation (to begin)

```console
python -m pip install --user -r requirements.txt
```


## Check test coverage : 

We use [coverage.py](https://coverage.readthedocs.io/en/stable/) for measuring code coverage of Python programs. It monitors your program, noting which parts of the code have been executed, then analyzes the source to identify code that could have been executed but was not.

### Configuration 

We use this simple option 

```console
coverage run --source=acm -m pytest tests 
```

### Script for acm

```console
acm_tests.sh
```


`coverage` provides also pretty HTML ouput page by module that indicate zone coverage. 
Open file 

```console
quality/html_coverage/index.html
```
with a web navigator.

### Launch only one test

During a debugging phase, it is practical to launch only the test you are working on, this is possible with the following command

```console
pytest tests/path/to/my/test::test_to_fix
```


## Check code, static analysis

We use [pylint](https://www.pylint.org/) as static code analysis to check coding standard (PEP8) and as error detector. List of [pylint message](http://pylint-messages.wikidot.com/all-codes), 5 kind of message:

* **Fatal** : prevents Pylint from parsing (syntax error, missing import, etc.).
* **Error** : logical error or incorrect code.
* **Warning** : potential problem, but not necessarily an error.
* **Refactoring** : possible improvements (complexity, duplication, etc.).
* **Convention** : style, naming, formatting.

### Configuration 

`pylint` is highly configurable via its configuration file: 

```console
quality/pylint.conf
```

### Script for acm

```console
acm_lint.sh
```

The analysis results are displayed in the console and also saved in the `report_pylint.txt` file.
The analysis concludes with an overall score out of 10. It is recommended to pay particular attention to error messages with:

```console
grep '\[E' report_pylint.txt
```

for a specific module

```console
grep 'acm/model/optimize' report_pylint.txt
```
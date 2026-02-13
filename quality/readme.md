# Code quality tools for developper

## Install initialize quality tools

```console
pip -e acm[testing,...]
source acm/quality/acm_quality_init.sh
```

Now you must have this acm scripts in your PATH
```console
$acm_
acm_clean_output.sh  acm_pylint_slow.sh   acm_tests.sh         
acm_lint.sh          acm_quality_init.sh  acm_type.sh          
```

## Quality tools in line command

* `acm_tests.sh` launches [pytest](https://docs.pytest.org/en/stable/) on `acm/tests/acm` directory and [coverage](https://coverage.readthedocs.io/en/latest/), a HTML report is generated. Return also the percent of code tested.
* `acm_lint.sh` launches [ruff](https://docs.astral.sh/ruff/) on `acm/acm` directory, a HTML report is generated
* `acm_type.sh` launches type checker [ty](https://docs.astral.sh/ty/) on `acm/acm`
* `acm_format_file.sh` applies formatter and sort the import for a specific file. The first use may result in many changes.
* `acm_lint_fix_file.sh` fixes some issues reported by `acm_lint.sh` for a specific file.



## Quality tools with VS Code

* python syntax,PEP/lint : with the [pluggin ruff from Astral Software](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff), you can format your code and import. ruff check your code and can propose you a fix.
    * select : View/problem to have a list of issue/warning detected by ruff
* check type/annotation : with the [pluggin ty from Astral Software](https://marketplace.visualstudio.com/items?itemName=astral-sh.ty)
    * Enable `SSH: perlmutter.nersc.gov`
    * As ruff issues are visible in View/problem window
* Copilot can generate tests
    * select module or function/method in module
    * right click
    * select: generate code/generate tests
    * a file of test is created in the same directory of module, so move it in `acm/tests/acm` directory


## Read HTML reports with VSCode

* Install the Live Server extension (search for Live Server in VS Code).
* Open your project folder.
* Right-click on an HTML file → “Open with Live Server” at the top
* The page will open in your browser.
 
## How implement tests

### With pytest framework

Here a [How-to guides](https://docs.pytest.org/en/stable/how-to/index.html) to begin with pytest.

Some important conventions with pytest:
* file of test must begin by `test_`, like
    * `test_xxx.py`
* the name of class of test must begin by `Test`, like
    * `class TestDatasetToDict:`
* the name of test function/method must begin by `test_`, like
    * `def test_init(path_file):`
    * `def test_get_cov(self):`

acm tests are in directory `acm/tests/acm`

### Launch only one test

During a debugging phase, it is practical to launch only the test you are working on, this is possible with the following command

```console
pytest tests/path/to/my/test::test_to_fix
```

### Data file for test

Some functions and methods require input files that are too large for Git. GitHub Releases let you host these files in your project, and pytest can automatically download them when tests start. Simply add your test files to a GitHub Release and list their names in your pytest configuration.

* GitHub Release for acm : [storage_data_tests](https://github.com/epaillas/acm/releases/tag/data_test_v1), used "modify" to add file
* Configuration file where the test file can be added to the list of files to download : `acm/tests/conftest.py`
* download directory `acm/tests/data` is defined by a environment variable `ACM_TEST_DATA`


## Pylint

You can also use [pylint](https://www.pylint.org/) as static code analysis to check coding standard (PEP8) and as error detector like [ruff](https://docs.astral.sh/ruff/) but slower. List of [pylint message](http://pylint-messages.wikidot.com/all-codes), 5 kinds of message:

* **Fatal** : prevents Pylint from parsing (syntax error, missing import, etc.).
* **Error** : logical error or incorrect code.
* **Warning** : potential problem, but not necessarily an error.
* **Refactoring** : possible improvements (complexity, duplication, etc.).
* **Convention** : style, naming, formatting.

**An interesting point** for evaluating the overall quality of the packaging is that the analysis concludes with an overall score out of 10.


### Configuration 

`pylint` is highly configurable via its configuration file: 

```console
quality/pylint.conf
```

### Script for acm

```console
acm_pylint_slow.sh
```

The analysis results are displayed in the console and also saved in the `pylint_report.txt` file.

## Others quality tools

* type checker [mypy](https://mypy-lang.org/), like `ty` but slower
* code formatter [black](https://black.readthedocs.io/en/stable/)
    * `ruff` has a formatter features 
* import formatter [isort](https://pycqa.github.io/isort/)
    * `ruff` format also import

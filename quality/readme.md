# Code quality tools for developper

## Install quality tools

```console
pip -e acm[testing,...]
```

## Quality tools

* `acm_tests.sh` launches [pytest](https://docs.pytest.org/en/stable/) on `acm/tests` directory and [coverage](https://coverage.readthedocs.io/en/latest/), a HTML report is generated
* `acm_lint.sh` launches [ruff](https://docs.astral.sh/ruff/) on `acm/acm` directory, a HTML report is generated
* `acm_type.sh` launches [mypy](https://mypy.readthedocs.io/en/stable/) on `acm/acm`


## Read HTML reports with VSCode

* Install the Live Server extension (search for Live Server in VS Code).
* Open your project folder.
* Right-click on an HTML file → “Open with Live Server” at the top
* The page will open in your browser.
 

## Launch only one test

During a debugging phase, it is practical to launch only the test you are working on, this is possible with the following command

```console
pytest tests/path/to/my/test::test_to_fix
```

## Pylint

We use [pylint](https://www.pylint.org/) as static code analysis to check coding standard (PEP8) and as error detector like [ruff](https://docs.astral.sh/ruff/) but slower. List of [pylint message](http://pylint-messages.wikidot.com/all-codes), 5 kinds of message:

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
acm_pylint_slow.sh report_
```

The analysis results are displayed in the console and also saved in the `pylint_report.txt` file.
The analysis concludes with an overall score out of 10. It is recommended to pay particular attention to error messages with:

```console
grep '\[E' pylint_report.txt
```

for a specific module

```console
grep 'acm/model/optimize' pylint_report.txt
```

## Others quality tools

* code formatter [black](https://black.readthedocs.io/en/stable/)
* import formatter [isort](https://pycqa.github.io/isort/)

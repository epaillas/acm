from acm.utils.decorators import temporary_class_state, require_nersc
from acm.utils.default import is_nersc


class TestDecorators:
    def __init__(self):
        self.val = 10
        self.str = "test"
        self.flag = True

    def change_attr(self):
        self.val = 0
        self.str = ""
        self.flag = False

    @temporary_class_state(str="", flag="")
    def change_only_val(self):
        self.change_attr()


def test_temporary_class_state():
    objtest = TestDecorators()
    objref = TestDecorators()
    objtest.change_only_val()
    assert objtest.val != objref.val
    assert objtest.str == objref.str
    assert objtest.flag == objref.flag
    objtest.change_attr()
    assert objtest.str != objref.str
    assert objtest.flag != objref.flag


@require_nersc(True)
def dummy_test():
    return True


def test_require_nersc():
    try:
        dummy_test()
        ret = True
    except Exception:
        ret = False
    assert is_nersc == ret



class BaseObservable:
    
    def __init__(self, data=None, covariance=None, theory=None):
        self.data = data
        self.covariance = covariance
        self.theory = theory
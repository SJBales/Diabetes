# Class to raise exception that the data has not been processed
class NotProcessedError(RuntimeError):
    pass


class ModelError(ValueError):
    pass

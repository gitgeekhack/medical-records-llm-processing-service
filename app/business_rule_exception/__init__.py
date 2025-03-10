from app.constant import ExceptionMessage


class MissingResponseListException(Exception):

    def __init__(self, message=ExceptionMessage.MISSING_RESPONSE_LIST_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

from app.constant import ExceptionMessage


class TextExtractionFailed(Exception):
    def __init__(self, message=ExceptionMessage.TEXTRACT_FAILED_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

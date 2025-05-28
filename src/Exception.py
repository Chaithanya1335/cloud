import sys

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb  = error_detail.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error Occured in Python Script Name {0} at line no {1} error is {2}".format(
        filename,exc_tb.tb_lineno,error
    )

    return error_message


class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(error)
        self.error_message = error_message_detail(error=error,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    




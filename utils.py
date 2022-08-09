import datetime
import time
import os

def current_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%y%m%d.%H%M%S")

def save_image(image, path=None, file_name=None):
    if not path:
        path = os.getcwd()
    if not file_name:
        timestamp = current_timestamp()
        file_name = "no_name_" + timestamp + '.tif'

    file_name = os.path.join(path, file_name)
    print(file_name)



if __name__ == '__main__':
    print(current_timestamp())
    save_image(image=1)
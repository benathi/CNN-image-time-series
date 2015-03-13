import os, inspect

def getCurrentPath():
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return current_folder_path

if __name__ == '__main__':
    print getCurrentPath()
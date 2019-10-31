import os
import shutil
import errno


def delete_directory_contents(rel_path):
    path = os.path.join(os.getcwd(), rel_path)
    print(f"Deleting contents of {path}.")
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


# if path = "folder/blah" or "folder/", will create folder.
def create_directory(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
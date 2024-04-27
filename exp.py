import shutil
import os

def create_folder_for_db(name) -> None:
    try:
        shutil.rmtree(name)
    except:
        pass
    finally:
        os.mkdir(name)


# create_folder_for_db('vector')
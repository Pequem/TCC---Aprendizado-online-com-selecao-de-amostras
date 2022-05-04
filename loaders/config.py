import json


class Loader:
    def __init__(self, path: str):
        self.path = path
    
    def valid(self, obj) -> bool:
        pass
        #if not obj['path']:
        #    raise Exception("Invalid config file")

    def read(self):
        try:
            file = open(self.path)

            data = json.load(file)

            file.close()

            self.valid(data)

            return data
        except:
            return False

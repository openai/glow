import json


class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        self.f_log = open(path, 'w')
        self.f_log.write(json.dumps(kwargs) + '\n')

    def log(self, **kwargs):
        self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()

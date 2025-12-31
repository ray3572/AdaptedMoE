class Logger():
    def __init__(self, path):
        self.path = path

    def add_log(self, log,is_show=False):
        with open(self.path, 'a') as f:
            if is_show:
                print(log)
            f.write(log)
            f.write('\n')

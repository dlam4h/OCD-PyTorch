import os


class LogWay(object):
    def __init__(self, log_file, first=True):
        self.log_file = log_file
        # if not os.path.exists(self.log_file):
        #     os.mkdir(self.log_file)
        self.n = 0
        self.first = first
        if self.first:
            self.head('the current process is {}.\n'.format(os.getpid()))

    def reset(self):
        pass

    def head(self, head_info):
        with open(self.log_file, 'w') as f:
            f.writelines(head_info)

    def add(self, log_info, print_flag=True):
        self.n += 1
        if print_flag:
            print(log_info, end='')

        with open(self.log_file, 'a') as f:
            f.writelines(log_info)

    def add_train(self, print_flag=True):
        pass

    def plot(self):
        pass
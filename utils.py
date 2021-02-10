import datetime
import time

class Timer:

    def __enter__(self):
        self.start = datetime.datetime.now()
        print('Starting {}...'.format(self.msg))
        return self

    def __exit__(self, *args):
        self.stop = datetime.datetime.now()
        self.elapsed = self.stop - self.start
        print('{} took {} seconds'.format( self.msg, self.elapsed.seconds ))


    def __init__(self, msg='operation'):
        self.msg = msg
        self.start = None
        self.end = None
        self.elapsed = None




def main():
    with Timer('stuff'):
        time.sleep(3)
        print('hey!')



if __name__ == '__main__':
    main()
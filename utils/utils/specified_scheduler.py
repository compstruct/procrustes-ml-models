
class lr_scheduled:
    def __init__(self, file_path='schedule.csv'):
        self.lines = None
        with open(file_path,'r') as f:
            self.lines = f.readlines()
    def get_lr(self, ep=0):
        return float(self.lines[ep].split()[0])

#lr_schedul = lr_scheduled('test_schedule.csv')
#print(lr_schedul.get_lr(100))

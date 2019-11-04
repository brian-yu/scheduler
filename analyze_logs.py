import os
from collections import defaultdict

from constants import Event

class LogAnalyzer:

    def __init__(self, scheduler_log_path):
        self.job_start_times = {}
        self.job_end_times = {}

        self.job_min_loss = {}
        self.job_max_acc = {}

        self.start_time = None
        self.end_time = None

        self.real_start_time = float('inf')
        self.real_end_time = 0



    def get_makespan(self):
        return self.end_time - self.start_time

    def get_real_makespan(self):
        return self.real_end_time - self.real_start_time

    def get_job_completion_times(self):
        completion_times = []
        for job in sorted(self.job_start_times.keys()):
            completion_times.append((job, self.job_end_times[job] - self.job_start_times[job]))
        return completion_times

    def update_job_start(self, job_name, time):
        if not self.start_time:
            self.start_time = time
        else:
            self.start_time = min(self.start_time, time)

        if job_name not in self.job_start_times:
            self.job_start_times[job_name] = time
            return

        self.job_start_times[job_name] = min(self.job_start_times[job_name], time)

    def update_job_end(self, job_name, time):
        if not self.end_time:
            self.end_time = time
        else:
            self.end_time = max(self.end_time, time)

        if job_name not in self.job_end_times:
            self.job_end_times[job_name] = time
            return

        self.job_end_times[job_name] = max(self.job_end_times[job_name], time)

    def add_log(self, filename):

        self.worker_save_restore_time = 0

        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                self.process_line(line)

        print("Time spent during save and restore")
        print(self.worker_save_restore_time)
                

    def process_line(self, line):
        tokens = line.rstrip("\n").split()
        job_name = tokens[0]
        event = Event(tokens[1])

        # print(self.worker_save_restore_time)

        if event == Event.TRAIN:
            action = tokens[2]
            time = float(tokens[3])

            if action == "START":
                self.update_job_start(job_name, time - self.worker_save_restore_time)
                self.start_time = min(self.start_time, time - self.worker_save_restore_time)
                self.real_start_time = min(self.real_start_time, time)
            elif action == "END":
                self.update_job_end(job_name, time - self.worker_save_restore_time)
                self.end_time = max(self.end_time, time - self.worker_save_restore_time)
                self.real_end_time = max(self.real_end_time, time)

        if event == Event.SAVE:
            action = tokens[2]
            time = float(tokens[3])

            if action == "START":
                self.worker_save_start = time
            elif action == "END":
                self.worker_save_restore_time += time - self.worker_save_start

        if event == Event.RESTORE:
            action = tokens[2]
            time = float(tokens[3])

            if action == "START":
                self.worker_restore_start = time
            elif action == "END":
                self.worker_save_restore_time += time - self.worker_restore_start

        elif event == Event.VAL_ACC:
            acc = float(tokens[2])
            if job_name not in self.job_max_acc:
                self.job_max_acc[job_name] = acc
            self.job_max_acc[job_name] = max(self.job_max_acc[job_name], acc)

        elif event == Event.VAL_LOSS:
            loss = float(tokens[2])
            if job_name not in self.job_min_loss:
                self.job_min_loss[job_name] = loss
            self.job_min_loss[job_name] = min(self.job_min_loss[job_name], loss)



def main():
    log_folder = './log_folder'

    analyzer = LogAnalyzer(os.path.join(log_folder, 'scheduler_log'))

    for filename in os.listdir(log_folder):
        if filename.startswith("worker"): 
            path = os.path.join(log_folder, filename)
            print(path)
            analyzer.add_log(path)


    print("Real makespan")
    print(analyzer.get_makespan())
    print("Makespan discounting save and restore times")
    print(analyzer.get_makespan())
    print("Job completion times")
    print(analyzer.get_job_completion_times())
    print("Job max accuracies")
    print(analyzer.job_max_acc)
    print("Job min losses")
    print(analyzer.job_min_loss)



if __name__ == "__main__":
    main()
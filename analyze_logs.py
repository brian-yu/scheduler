import os
from collections import defaultdict

from constants import Event

class LogAnalyzer:

    def __init__(self, log_folder):

        self.log_folder = log_folder

        self.job_start_times = {}
        self.job_end_times = {}

        self.job_min_loss = {}
        self.job_max_acc = {}

        self.job_executable = {}
        self.job_epochs = {}

        self.job_acc = defaultdict(list)
        self.job_loss = defaultdict(list)

        self.start_time = float('inf')
        self.end_time = 0

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
        if job_name not in self.job_start_times:
            self.job_start_times[job_name] = time
            return

        self.job_start_times[job_name] = min(self.job_start_times[job_name], time)

    def update_job_end(self, job_name, time):
        if job_name not in self.job_end_times:
            self.job_end_times[job_name] = time
            return

        self.job_end_times[job_name] = max(self.job_end_times[job_name], time)

    def add_logs(self):
        for filename in os.listdir(self.log_folder):
            if filename.startswith("worker"): 
                path = os.path.join(log_folder, filename)
                print(path)
                self.add_log(path)
            elif filename.startswith("scheduler"):
                path = os.path.join(log_folder, filename)
                print(path)
                self.add_scheduler_log(path)

        for arr in self.job_acc.values():
            arr.sort()

        for arr in self.job_loss.values():
            arr.sort()

    def add_scheduler_log(self, filename):
        with open(filename) as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
            for line in lines:
                job_name, num_epochs, executable = line.split()
                self.job_executable[job_name] = executable
                self.job_epochs[job_name] = num_epochs

    def add_log(self, filename):

        self.worker_save_restore_time = 0

        with open(filename) as f:
            lines = f.readlines()
            prev_time = None
            for line in lines:
                prev_time = self.process_line(line, prev_time)

    def process_line(self, line, prev_time = None):
        tokens = line.rstrip("\n").split()
        job_name = tokens[0]
        event = Event(tokens[1])

        # print(self.worker_save_restore_time)
        time = None

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

            if not prev_time:
                raise Exception("prev_time should not be None")
            self.job_acc[job_name].append((prev_time, acc))

        elif event == Event.VAL_LOSS:
            loss = float(tokens[2])
            if job_name not in self.job_min_loss:
                self.job_min_loss[job_name] = loss
            self.job_min_loss[job_name] = min(self.job_min_loss[job_name], loss)
            if not prev_time:
                raise Exception("prev_time should not be None")
            self.job_loss[job_name].append((prev_time, loss))

        return time

    def info(self):
        print("Real makespan")
        print(f"{' ' * 4}{self.get_real_makespan()}")
        print("Makespan discounting save and restore times")
        print(f"{' ' * 4}{self.get_makespan()}")
        print("Job Name\tEpochs\tExecutable\t\tCompletion Time\t\tBest Acc.\tBest Loss")
        for job in sorted(self.job_epochs.keys(), key=lambda job: int(job.split('_')[1])):
            num_epochs = self.job_epochs[job]
            executable = self.job_executable[job]
            completion_time = self.job_end_times[job] - self.job_start_times[job]
            best_acc = self.job_max_acc[job]
            best_loss = self.job_min_loss[job]
            print(f"{job}\t\t{num_epochs}\t{executable:20.20}\t{completion_time:.2f}\t\t\t{best_acc:.4f}\t\t{best_loss:.4f}")

        print("Job accuracies.")
        for job in sorted(self.job_epochs.keys(), key=lambda job: int(job.split('_')[1])):
            print(job)
            print(self.job_acc[job])

        print("Job losses.")
        for job in sorted(self.job_epochs.keys(), key=lambda job: int(job.split('_')[1])):
            print(job)
            print(self.job_loss[job])




def main():
    log_folder = './log_folder'

    analyzer = LogAnalyzer(log_folder)

    

    analyzer.info()



if __name__ == "__main__":
    main()
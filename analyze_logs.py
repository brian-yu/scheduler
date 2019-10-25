import os




class LogAnalyzer:

    def __init__(self):
        self.job_start_times = {}
        self.job_end_times = {}

        self.start_time = None
        self.end_time = None

    def get_makespan(self):
        return self.end_time - self.start_time

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
        with open(filename) as f:
            lines = [self.parse_line(line) for line in f.readlines()]
            # print(lines)

            for job_name, action, time in lines:
                if action == "start":
                    self.update_job_start(job_name, time)
                elif action == "end":
                    self.update_job_end(job_name, time)
                

    def parse_line(self, line):
        job_name, action, time = line.rstrip("\n").split(",")
        return job_name, action, float(time)



def main():
    log_folder = './log_folder'

    analyzer = LogAnalyzer()

    for filename in os.listdir(log_folder):
        if filename.startswith("worker"): 
            path = os.path.join(log_folder, filename)
            print(path)
            analyzer.add_log(path)

    print(analyzer.get_makespan())
    print(analyzer.get_job_completion_times())



if __name__ == "__main__":
    main()
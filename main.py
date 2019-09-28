from worker import Worker

import multiprocessing
import time

def do_stuff(n): 
    # time.sleep(duration)
    res = []
    for i in range(n):
        res.append(n**2)

  
  
if __name__ == "__main__": 
    start = time.time()
    # creating processes 
    workers = [multiprocessing.Process(target=do_stuff, args=(100000000, )) for _ in range(7)]
  
    for worker in workers:
        worker.daemon = True # Kill child process when master
        worker.start()

    # Communicate with workers here.








    for worker in workers:
        worker.join()

    end = time.time()
  
    # both processes finished 
    print(f"Done in {end - start}s!") 
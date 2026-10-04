"""Minimal file-based GPU job queue.

queue/pending/<job>.sh  -> claimed atomically (rename) into queue/running/ by one worker per GPU,
then moved to queue/done/ (exit 0) or queue/failed/. Each job script receives the GPU id as $1.
Run:  python run_queue.py 0 1 2 3 4 5 6 7
The queue directory is $RUN_QUEUE_DIR (default ~/repro/queue); create a file named STOP in it to stop the
workers once their current jobs finish.
"""
import os, sys, time, subprocess, threading, datetime

Q = os.path.expanduser(os.environ.get('RUN_QUEUE_DIR', '~/repro/queue'))
for d in ('pending', 'running', 'done', 'failed'):
    os.makedirs(os.path.join(Q, d), exist_ok=True)


def log(msg):
    with open(os.path.join(Q, 'queue.log'), 'a') as f:
        f.write(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n')


def claim():
    for name in sorted(os.listdir(os.path.join(Q, 'pending'))):
        try:
            os.rename(os.path.join(Q, 'pending', name), os.path.join(Q, 'running', name))
            return name
        except FileNotFoundError:
            continue  # claimed by another worker
    return None


def worker(gpu):
    while not os.path.exists(os.path.join(Q, 'STOP')):
        name = claim()
        if name is None:
            time.sleep(20)
            continue
        log(f'START gpu{gpu} {name}')
        t0 = time.time()
        rc = subprocess.call(['bash', os.path.join(Q, 'running', name), str(gpu)])
        dst = 'done' if rc == 0 else 'failed'
        os.rename(os.path.join(Q, 'running', name), os.path.join(Q, dst, name))
        log(f'{dst.upper()} gpu{gpu} {name} rc={rc} {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    gpus = [int(g) for g in sys.argv[1:]]
    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

import psutil
import os
import time

def kill_all_python_processes():
    current_pid = os.getpid()
    killed_count = 0

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'python' in proc.info['name'].lower() and proc.info['pid'] != current_pid:
                psutil.Process(proc.info['pid']).kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return killed_count

if __name__ == "__main__":
    count = kill_all_python_processes()
    print(f"Killed {count} Python process{'es' if count != 1 else ''}. Exiting self.")
    time.sleep(1)  # Allow time to print before self-termination
    os.kill(os.getpid(), 9)

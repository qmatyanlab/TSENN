import subprocess

if __name__ == "__main__":
    for gpu_id in range(4):
        subprocess.Popen(["python", "optuna_worker.py", str(gpu_id)])

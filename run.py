import subprocess
import json, psutil, datetime, sys, time
import argparse

if __name__ == "__main__":
    proc = subprocess.Popen("python main.py "+ " ".join(sys.argv[3:]), stdout=open(sys.argv[1], "w"))

    print("PID:", proc.pid)
    with open(sys.argv[2], 'w') as f:
        while True:
            try:
                res = (datetime.datetime.now().isoformat(),
                                psutil.Process(int(proc.pid)).memory_info()._asdict())

                f.write(str(res[1]["rss"]) + '\n')
            except:
                print("Finished!")
                break
            time.sleep(1)

    print("Return code:", proc.wait())

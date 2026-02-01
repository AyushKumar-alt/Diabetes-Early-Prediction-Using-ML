"""
Safe runner for `tune_calibrated.py`.
- Sets thread/OMP limits to reduce memory pressure
- Captures stdout/stderr to a timestamped log
- Retries once with fewer iterations on unexpected failure

Usage:
    python run_tuning_safe.py [n_iter]

"""
import os
import sys
import time
import subprocess

def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    # Limit threads to reduce OOM risk
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    logfile = os.path.abspath(f'tuning_safe_{timestamp}.log')

    cmd = [sys.executable, '-m', 'diabetes_risk_app.project.tune_calibrated']
    env = os.environ.copy()
    env['TUNE_N_ITER'] = str(n_iter)

    print(f'Running tuning with n_iter={n_iter}. Log: {logfile}')
    with open(logfile, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        ret = proc.wait()

    if ret == 0:
        print('Tuning completed successfully. See log:', logfile)
        return 0

    # On failure, retry with fewer iterations
    print('Tuning failed (exit code %d). Retrying with smaller budget (n_iter=8)...' % ret)
    logfile2 = os.path.abspath(f'tuning_safe_retry_{timestamp}.log')
    cmd2 = cmd.copy()
    env2 = env.copy()
    env2['TUNE_N_ITER'] = '8'
    with open(logfile2, 'w', encoding='utf-8') as f:
        proc2 = subprocess.Popen(cmd2, stdout=f, stderr=subprocess.STDOUT, env=env2)
        ret2 = proc2.wait()

    if ret2 == 0:
        print('Retry succeeded. See log:', logfile2)
        return 0

    print('Retry failed (exit code %d). Check logs: %s and %s' % (ret2, logfile, logfile2))
    return ret2

if __name__ == '__main__':
    sys.exit(main())

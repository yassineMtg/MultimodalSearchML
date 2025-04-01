# launch_tensorboard.py
import os
import subprocess

def launch_tensorboard():
    # Automatically detects the right path
    project_root = os.path.dirname(os.path.abspath( ".."))
    logdir = os.path.join(project_root, "artifacts")
    
    print(f"🚀 Launching TensorBoard on yassine-ubuntu-desktop:6006")
    print(f"👉 Access it via http://yassine-ubuntu-desktop:6006 from any device on your network")
    print(f"✅ Logs directory: {logdir}")

    # Launch TensorBoard bound to 0.0.0.0
    subprocess.run(["tensorboard", "--logdir", logdir, "--host", "0.0.0.0"])

if __name__ == "__main__":
    launch_tensorboard()

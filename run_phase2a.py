import subprocess
import sys
import os
import shutil

def run_colab():
    session = "phase2a"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    inf_dir = os.path.join(base_dir, "inference")
    
    print(f"1. Provisioning Colab GPU session '{session}'...")
    subprocess.run(["colab", "new", "-s", session, "--gpu", "T4"], check=True)
    
    zip_path = os.path.join(base_dir, "geodiff3d_payload.zip")
    print("Zipping payload...")
    subprocess.run(["zip", "-r", zip_path, "core", "inference"], cwd=base_dir, check=True)
    
    try:
        print("2. Uploading payload zip...")
        subprocess.run(["colab", "upload", "-s", session, zip_path, "/content/geodiff3d_payload.zip"], check=True)
        
        print("3. Executing GPU Pipeline...")
        runner_path = os.path.join(base_dir, "runner.sh")
        subprocess.run(["colab", "exec", "-s", session, "-f", runner_path], check=True)
        
        print("4. Downloading outputs...")
        out_zip = "/content/output.zip"
        subprocess.run(["colab", "download", "-s", session, out_zip, "-o", base_dir], check=True)
        
        print("Unzipping outputs locally...")
        subprocess.run(["unzip", "-o", "output.zip", "-d", "geodiff3d_output_tmp"], cwd=base_dir, check=True)
        # Move the output folder to inference/output
        if os.path.exists(os.path.join(base_dir, "inference", "output")):
            shutil.rmtree(os.path.join(base_dir, "inference", "output"))
        shutil.move(os.path.join(base_dir, "geodiff3d_output_tmp", "content", "inference", "output"), os.path.join(base_dir, "inference", "output"))
        shutil.rmtree(os.path.join(base_dir, "geodiff3d_output_tmp"))
        
    finally:
        print("5. Stopping Colab session...")
        subprocess.run(["colab", "stop", "-s", session], check=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)

if __name__ == "__main__":
    run_colab()

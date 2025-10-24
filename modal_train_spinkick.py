"""
Modal training for ACTUAL spinkick motion tracking!
"""

import modal

app = modal.App("g1-spinkick-real")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "git", "wget", "unzip", "libgl1-mesa-glx", "libglib2.0-0", 
        "libegl1-mesa", "xvfb"
    ])
    .pip_install([
        "torch", "numpy", "wandb", "tyro", "scipy", "tqdm", 
        "gymnasium", "mujoco>=3.1.0", "uv"
    ])
    .env({"MUJOCO_GL": "egl"})
)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_spinkick_real(
    num_envs: int = 256,
    max_iterations: int = 2000,
    registry_name: str = "jonathan-muhire-oklahoma-christian-university/spinkick-motion-registry/spinkick_safe_motion:latest"
):
    """Train REAL spinkick motion tracking!"""
    import subprocess
    import time
    import os
    
    print(f"🚀 REAL Spinkick Training!")
    print(f"💪 A100 GPU - {num_envs} envs - {max_iterations} iterations")
    print(f"📊 Registry: {registry_name}")
    
    # Clone repos
    subprocess.run(["mkdir", "-p", "/app"], check=True)
    subprocess.run([
        "git", "clone", "https://github.com/mujocolab/mjlab", "/app/mjlab"
    ], check=True)
    subprocess.run([
        "git", "clone", "https://github.com/mujocolab/g1_spinkick_example", "/app/g1_spinkick_example"
    ], check=True)
    
    os.chdir("/app/g1_spinkick_example")
    
    # Install dependencies using pip instead of uv to avoid conflicts
    subprocess.run(["pip", "install", "-e", "../mjlab"], check=True)
    subprocess.run(["pip", "install", "-e", "."], check=True)
    
    # Setup WandB
    try:
        import wandb
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project="g1-spinkick-training",
            name=f"spinkick_real_{num_envs}envs_{max_iterations}iter",
            tags=["modal", "A100", "spinkick", "real-motion"]
        )
        print(f"✅ WandB: {wandb.run.url}")
    except Exception as e:
        print(f"⚠️ WandB setup failed: {e}")
    
    # Run SPINKICK training with motion registry
    print("🌟 Starting SPINKICK motion tracking training...")
    
    cmd = [
        "python", "train.py",
        "Mjlab-Spinkick-Unitree-G1",  # This is the actual spinkick task!
        "--registry-name", registry_name,
        "--env.scene.num-envs", str(num_envs),
        "--agent.max-iterations", str(max_iterations),
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        print(f"✅ SPINKICK training completed in {duration/60:.2f} minutes!")
        print("Last 1000 chars of output:")
        print(result.stdout[-1000:])
        return {"status": "success", "duration_minutes": duration/60}
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ Training failed after {duration/60:.2f} minutes")
        print("Stdout:", e.stdout[-1000:] if e.stdout else "No stdout")
        print("Stderr:", e.stderr[-1000:] if e.stderr else "No stderr")
        return {"status": "error", "duration_minutes": duration/60, "error": str(e)}

@app.local_entrypoint()
def main(
    num_envs: int = 256,
    max_iterations: int = 2000,
):
    """Launch REAL spinkick training!"""
    print(f"🚀 Launching REAL G1 Spinkick Training!")
    print(f"🎯 This will teach the robot to do actual spin kicks!")
    print(f"💰 Estimated cost: ~${max_iterations/10000 * 2.5:.2f}")
    
    result = train_spinkick_real.remote(num_envs, max_iterations)
    print(f"Result: {result}")
    return result
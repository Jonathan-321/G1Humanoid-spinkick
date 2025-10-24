"""
Modal app for training G1 spin kick with mjlab on GPU.

This script provides:
- GPU-accelerated training on Modal
- Cost monitoring and warnings
- WandB integration
- Error handling and logging
- Configurable environment counts for testing

Usage:
  modal run modal_train.py --num-envs 256 --max-iterations 1000 --wandb-project "g1-spinkick-test"
"""

import os
import time
from pathlib import Path
from typing import Optional

import modal

# Create Modal app
app = modal.App("g1-spinkick-training")

# Define the compute environment
# A100 for faster training (more expensive but much faster)
gpu_config = "A100"  # A100 for optimal performance

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "git", 
        "wget", 
        "unzip",
        "libgl1-mesa-glx",  # OpenGL for rendering
        "libglib2.0-0",     # Required for MuJoCo
        "libegl1-mesa",     # EGL for headless rendering
        "xvfb",             # Virtual display
    ])
    .pip_install([
        "uv",
        "torch>=2.0.0",
        "numpy",
        "wandb",
        "tyro",
        "scipy",
        "tqdm",
        "gymnasium",
        "mujoco>=3.1.0",
    ])
    .env({"MUJOCO_GL": "egl"})  # Headless rendering
)

# Mount local code
local_mount = modal.Volume.from_name("spinkick-code", create_if_missing=True)

@app.function(
    image=image,
    gpu=gpu_config,
    mounts=[local_mount],
    timeout=3600,  # 1 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],  # WandB API key
)
def train_spinkick(
    num_envs: int = 256,
    max_iterations: int = 1000,
    wandb_project: str = "g1-spinkick-test",
    wandb_entity: Optional[str] = None,
    registry_name: Optional[str] = None,
    gpu_type: str = "T4",
    dry_run: bool = False,
):
    """
    Train G1 spin kick on Modal GPU.
    
    Args:
        num_envs: Number of parallel environments (start with 256 for testing)
        max_iterations: Training iterations (1000 for quick test, 20000 for full)
        wandb_project: WandB project name
        wandb_entity: WandB entity/organization (optional)
        registry_name: Motion registry name (optional, will create default)
        gpu_type: GPU type for cost tracking
        dry_run: If True, just setup and validate without training
    """
    import subprocess
    import sys
    from datetime import datetime
    
    # Cost warning based on GPU type and iterations
    cost_estimates = {
        "T4": 0.50,     # per hour
        "A10G": 1.10,   # per hour  
        "A100": 2.50,   # per hour
    }
    
    estimated_hours = max_iterations / 10000  # Rough estimate: 10k iterations per hour
    estimated_cost = cost_estimates.get(gpu_type, 1.0) * estimated_hours
    
    print(f"\n🚨 COST WARNING 🚨")
    print(f"GPU Type: {gpu_type}")
    print(f"Estimated runtime: {estimated_hours:.2f} hours")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Environments: {num_envs}")
    print(f"Max iterations: {max_iterations}")
    
    if estimated_cost > 5.0:
        print(f"⚠️  HIGH COST WARNING: This run may cost ${estimated_cost:.2f}")
        print("Consider reducing --max-iterations or --num-envs for testing")
    
    if dry_run:
        print("🧪 DRY RUN - Exiting without training")
        return {"status": "dry_run", "estimated_cost": estimated_cost}
    
    # Change to app directory
    os.chdir("/app")
    
    # Install local dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.run(["uv", "add", "--editable", "./mjlab"], check=True)
        subprocess.run(["uv", "add", "--editable", "."], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return {"status": "error", "error": "dependency_install_failed"}
    
    # Setup WandB
    print("🔧 Setting up WandB...")
    import wandb
    
    # Initialize WandB with error handling
    try:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        
        # Create run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"spinkick_test_{num_envs}envs_{timestamp}"
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            tags=["modal", "test", gpu_type.lower()],
            config={
                "num_envs": num_envs,
                "max_iterations": max_iterations,
                "gpu_type": gpu_type,
                "platform": "modal",
            }
        )
        print(f"✅ WandB initialized: {wandb.run.url}")
        
    except Exception as e:
        print(f"⚠️  WandB setup failed: {e}")
        print("Continuing without WandB logging...")
    
    # Create default registry name if not provided
    if not registry_name:
        registry_name = f"{wandb_entity or 'test'}/g1-spinkick/mimickit_spinkick_safe"
    
    print(f"🏃 Starting training...")
    print(f"Registry: {registry_name}")
    
    # Build training command
    cmd = [
        "uv", "run", "train.py",
        "Mjlab-Spinkick-Unitree-G1",
        "--registry-name", registry_name,
        f"--env.scene.num-envs", str(num_envs),
        f"--agent.max-iterations", str(max_iterations),
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run training with error handling
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        actual_cost = cost_estimates.get(gpu_type, 1.0) * (duration / 3600)
        
        print(f"✅ Training completed!")
        print(f"Duration: {duration/60:.2f} minutes")
        print(f"Actual cost: ${actual_cost:.2f}")
        
        return {
            "status": "success", 
            "duration_minutes": duration/60,
            "actual_cost": actual_cost,
            "wandb_url": wandb.run.url if 'wandb' in locals() and wandb.run else None
        }
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        actual_cost = cost_estimates.get(gpu_type, 1.0) * (duration / 3600)
        
        print(f"❌ Training failed!")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        print(f"Duration: {duration/60:.2f} minutes")
        print(f"Cost incurred: ${actual_cost:.2f}")
        
        return {
            "status": "error", 
            "error": str(e),
            "duration_minutes": duration/60,
            "cost_incurred": actual_cost
        }

@app.local_entrypoint()
def main(
    num_envs: int = 256,
    max_iterations: int = 1000, 
    wandb_project: str = "g1-spinkick-test",
    wandb_entity: Optional[str] = None,
    registry_name: Optional[str] = None,
    gpu_type: str = "T4",
    dry_run: bool = False,
):
    """
    Local entrypoint for Modal training.
    
    Examples:
        # Test run (cheap)
        modal run modal_train.py --num-envs 256 --max-iterations 1000
        
        # Dry run (no cost)
        modal run modal_train.py --dry-run
        
        # Full training (expensive!)
        modal run modal_train.py --num-envs 4096 --max-iterations 20000 --gpu-type A100
    """
    print(f"🚀 Launching G1 Spinkick training on Modal...")
    
    result = train_spinkick.remote(
        num_envs=num_envs,
        max_iterations=max_iterations,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        registry_name=registry_name,
        gpu_type=gpu_type,
        dry_run=dry_run,
    )
    
    print(f"\n📊 Training Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
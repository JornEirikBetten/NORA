# Utils 
import os 
import sys 
import tqdm 
path_to_here = os.getcwd() 

# XLA flags
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=false "
)

# Plotting 
import matplotlib.pyplot as plt 
# Setting ggplot style 
plt.style.use("ggplot")

# Numpy & PyTorch  
import numpy as np 
import torch 
import torchvision 

# JAX 
import jax 
import jax.numpy as jnp
import flax 
from flax.training import train_state, checkpoints 
import optax 


# Source code 
import src 
from src import train_and_evaluate


batch_size = 512
mnist_data = src.MNISTData(batch_size)
images, labels = next(iter(mnist_data.train_loader))

model = src.DenseClassifier(num_hidden=64, num_outputs=10)
rng = jax.random.PRNGKey(12)
rng, init_rng = jax.random.split(rng, 2)
params = model.init(init_rng, images)
optimizer = optax.adam(learning_rate=1e-3)

model_state = train_state.TrainState.create(apply_fn=model.apply, 
                                            params=params, 
                                            tx=optimizer)

train_loader = mnist_data.train_loader
test_loader = mnist_data.test_loader
num_epochs = 50

trained_model_state, training_accuracy, testing_accuracy = train_and_evaluate(model_state, train_loader, test_loader, num_epochs)
checkpoints.save_checkpoint(ckpt_dir=path_to_here + '/saved_models/',  # Folder to save checkpoint in
                            target=trained_model_state,  # What to save. To only save parameters, use model_state.params
                            step=0, 
                            prefix='mnist',  # Checkpoint file name prefix
                            overwrite=True   # Overwrite existing checkpoint files
                           )

loaded_model_state = checkpoints.restore_checkpoint(
                                             ckpt_dir=path_to_here + '/saved_models/',   # Folder with the checkpoints
                                             target=model_state,   # (optional) matching object to rebuild state in
                                             prefix='mnist'  # Checkpoint file name prefix
                                            )

print(src.evaluate(loaded_model_state, test_loader))
fig = plt.figure() 
plt.plot([i+1 for i in range(num_epochs)], training_accuracy, label="training accuracy", color="black", linestyle="solid")
plt.plot([i+1 for i in range(num_epochs)], testing_accuracy, label="validation accuracy", color="black", linestyle="dashed")
plt.xlabel("epoch")
plt.ylabel("accuracy fraction")
plt.legend()
plt.savefig("training_mnist.pdf", format="pdf", bbox_inches="tight")

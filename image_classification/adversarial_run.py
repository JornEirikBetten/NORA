# Utils 
import os 
import sys 
import tqdm 
path_to_here = os.getcwd()
fig_path = path_to_here + "/figures/"
if not os.path.exists(fig_path): 
    os.makedirs(fig_path)
    
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

# Predict and loss
def loss_function(state, params, image, label): 
    logits = state.apply_fn(params, image)
    label=jnp.array([label])
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, label).mean()
    pred_labels = logits.argmax(axis=-1)
    accuracy = (pred_labels == label) 
    return loss, accuracy

def generate_adversarial(state, params, images, labels, epsilon):
    grad_image = jax.grad(loss_function, 
                          argnums=2, 
                          has_aux=True)
    gradient, _ = grad_image(state, params, images, labels)
    signed_gradient = jnp.sign(gradient)
    return images + epsilon *  signed_gradient
    


def perturb_and_check(state, params, batch, epsilon):
    images, labels = batch 
    num_points = images.shape[0]
    accuracy = 0 
    adversarial_accuracy = 0 
    logits = state.apply_fn(params, images)
    acc= (logits.argmax(-1) == labels).mean()
    accuracy += num_points*acc 
    adversarials = jax.vmap(generate_adversarial, in_axes=(None, None, 0, 0, None))(state, params, images, labels, epsilon)
    logits = state.apply_fn(params, adversarials)
    acc = (logits.argmax(-1) == labels).mean() 
    adversarial_accuracy += acc*num_points
        

    return accuracy, adversarial_accuracy

def plot_grid(images, adversarials, fig_path): 
    fig, axs = plt.subplots(5, 2, figsize=(6, 15))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, image in enumerate(images): 
        if i == 0: 
            axs[i][0].set_title("Originals")
            axs[i][1].set_title("Adversarials")
        axs[i][0].imshow(image, cmap="binary_r")
        axs[i][0].axis("off")
        axs[i][1].imshow(adversarials[i], cmap="binary_r")
        axs[i][1].axis("off") 
    plt.savefig(fig_path + "image_adversarial.pdf", format="pdf", bbox_inches="tight")
    plt.close() 
    return 0

def perturb_and_plot(state, params, batch, epsilon, fig_path): 
    images, labels = batch 
    num_points = images.shape[0]
    accuracy = 0 
    adversarial_accuracy = 0 
    logits = state.apply_fn(params, images)
    accuracies = logits.argmax(-1) == labels 
    acc = (logits.argmax(-1) == labels).mean()
    accuracy += num_points*acc 
    adversarials = jax.vmap(generate_adversarial, in_axes=(None, None, 0, 0, None))(state, params, images, labels, epsilon)
    logits = state.apply_fn(params, adversarials)
    adversarial_accuracies = logits.argmax(-1) == labels 
    acc = (logits.argmax(-1) == labels).mean() 
    adversarial_accuracy += num_points*acc 
    images_ = []; adversarials_ = []
    count = 0  
    while len(images_) < 5 or count > 512: 
        if accuracies[count]==1 and adversarial_accuracies[count]==0: 
            images_.append(images[count, 0, :, :]); adversarials_.append(adversarials[count, 0, :, :])
        count += 1 
    fig_path = fig_path + f"epsilon_{epsilon:.3f}_"
    plot_grid(images_, adversarials_, fig_path)
    return accuracy, adversarial_accuracy
    

jitted_perturb_and_check = jax.jit(perturb_and_check)
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



state = checkpoints.restore_checkpoint(
                                    ckpt_dir=path_to_here + '/saved_models/',   # Folder with the checkpoints
                                    target=model_state,   # (optional) matching object to rebuild state in
                                    prefix='mnist'  # Checkpoint file name prefix
                                    )

num_rounds = 1001
epsilons = np.linspace(0.001, 1.001, num_rounds)
accuracies = np.zeros(num_rounds)
adversarial_accuracies = np.zeros(num_rounds)
batch = next(iter(mnist_data.test_loader))
for i, epsilon in enumerate(epsilons):
    plot_ = False 
    total_acc = 0; total_adv_acc = 0  
    if epsilon == epsilons[79] or epsilon == epsilons[159] or epsilon == epsilons[329] or epsilon == epsilons[899]: 
        plot_ = True 
    for j, batch in enumerate(mnist_data.test_loader):
        if plot_ and j == 10: 
            accuracy, adversarial_accuracy = perturb_and_plot(state, state.params, batch, epsilon, fig_path)
        else:  
            accuracy, adversarial_accuracy = jitted_perturb_and_check(state, state.params, batch, epsilon)
        total_acc += accuracy; total_adv_acc += adversarial_accuracy
    accuracies[i] = total_acc / 10000; adversarial_accuracies[i] = total_adv_acc / 10000 
    print(f"Adversarial accuracy given eps={epsilon:.3f}: {total_adv_acc/10000: .2f}")
plt.figure()    
plt.plot(epsilons, accuracies, label="original classification", color="black", linestyle="dashed")
plt.plot(epsilons, adversarial_accuracies, label="classification after adversarial attack", color="black", linestyle="solid")
plt.xlabel("magnitude of perturbation, $\epsilon$")
plt.ylabel("accuracy fraction")
plt.legend()
plt.savefig("dependence_on_epsilon.pdf", format="pdf", bbox_inches="tight")
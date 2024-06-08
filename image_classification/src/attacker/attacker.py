import jax 
import flax 
import jax.numpy as jnp 

def modified_loss(image, parameters, target): 
    


def FGSM_perturbation(x, target, L, parameters, epsilon): 
    grad_fn = jax.grad(L, argnums=1)
    gradient = grad_fn(parameters, x, target)
    L_inf_gradient = jnp.sign(gradient)
    adversarial_example = x + epsilon * L_inf_gradient 
    return adversarial_example 


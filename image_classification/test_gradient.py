import jax 
import jax.numpy as jnp 

def loss(x, y): 
    return jnp.sum(jnp.abs(x-y))  

x = jnp.array([1., 2., 3., 4., 5.])
y = jnp.array([2., 3., 5., 1., 2.])

gradient_x = jax.value_and_grad(loss, 
                                argnums=0)

print(gradient_x(x, y))

gradient_y = jax.value_and_grad(loss, argnums=1)
print(gradient_y(x, y))


import jax 
import jax.numpy as jnp 
import numpy as np 
import optax 

# Predict and loss
def loss_function(state, params, batch): 
    images, labels = batch 
    logits = state.apply_fn(params, images)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    pred_labels = logits.argmax(axis=-1)
    accuracy = (pred_labels == labels).mean()
    return loss, accuracy


@jax.jit 
def train_step(state, batch): 
    grad_fn = jax.value_and_grad(loss_function, 
                                 argnums=1, 
                                 has_aux=True)
    (loss, acc), grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc

def evaluate(state, loader):
    accuracy = 0
    count = 0   
    for batch in loader:
        num_points = batch[0].shape[0] 
        count += num_points
        _, acc = loss_function(state, state.params, batch)
        accuracy += num_points*acc 
    return accuracy/count

def train_and_evaluate(state, train_loader, test_loader, num_epochs): 
    training_accuracy = np.zeros(num_epochs)
    testing_accuracy = np.zeros(num_epochs)
    
    epochs = range(num_epochs)
    for epoch in epochs: 
        print(f"Epoch={epoch}")
        accuracy = 0
        count = 0  
        for batch in train_loader: 
            state, loss, acc = train_step(state, batch)
            num_points = batch[0].shape[0]
            count += num_points 
            accuracy += num_points*acc 
        training_accuracy[epoch] = accuracy/count 
        testing_accuracy[epoch] = evaluate(state, test_loader)
    return state, training_accuracy, testing_accuracy
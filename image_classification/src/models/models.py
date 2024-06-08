import jax 
import flax.linen as nn 


class DenseClassifier(nn.Module): 
    num_hidden: int 
    num_outputs: int 
    
    def setup(self): 
        self.input_layer = nn.Dense(2*self.num_hidden)
        self.hidden_layer = nn.Dense(self.num_hidden)
        self.output_layer = nn.Dense(self.num_outputs)
        
    @nn.compact 
    def __call__(self, x): 
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(self.input_layer(x))
        x = nn.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
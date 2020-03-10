from keras import layers, models, optimizers
from keras import backend as K 

# Create a class called Critic, whose object takes in the following parameters:
class Critic:
    """Critic (Value) Model."""

    def __init__(self,state_size,action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size(int): Dimension of each state
            action_size(int) : Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        # Build a critic (value) network that maps state and action pairs (Q_values)
        states = layers.Input(shape=(self.state_size,),name='states')
        actions = layers.Input(shape=(self.action_size,),name='actions')

        # Add hidden layers for the state pathway
        net_states = layers.Dense(units=16,kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)

        net_states = layers.Dense(units=32,kernel_regularizer=layers.regularizers.l2(1e-6))(net_states)

        # Add the hidden layers for the action pathway
        net_actions = layers.Dense(units=32,kernel_regularizer=layers.regularizers.l2(1e-6))(actions)
        net = layers.Add()([net_states,net_actions])
        net = layers.Activation("relu")(net)

        # Add the final output layer to produce the action values(Q-values):
        Q_values = layers.Dense(units=1,name='q_values',kernel_initializer=layers.initializers.RandomUniform(minval=-0.003,maxval=0.003))(net)
        
        # Create Keras model
        self.model = models.Model(inputs = [states,actions],outputs=Q_values)

        # Define the optimizer and compile a model for training with the built-in loss functions:
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer,loss='mse')

        # Compute the action gradients (the derivative of Q_values with resepct to actions)
        action_gradients = K.gradients(Q_values,actions)

        # Define an additional function to fetch the action gradients (to be used by the actor model)
        self.get_action_gradients = K.function(
            inputs = [*self.model.input, K.learning_phase()],
            outputs = action_gradients
        )

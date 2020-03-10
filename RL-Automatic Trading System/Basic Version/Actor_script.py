from keras import layers, models, optimizers
from keras import backend as K

# Create a class called Actor
# Whose project takes in the parameters of the state and action size

class Actor:
    #"""Actor (Policy) Model."""

    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    # Build a policy model that maps the states to actions, and start by defining the input layer
    def build_model(self):
        # Define the input layer
        states = layers.Input(shape=(self.state_size,),name='states')

        # Add hidden layers to the model.
        # There are two dense layers, each one follows by a batch normalization and activation layer
        # The dense layers are regularized.
        # The two layers have 16 and 32 hidden units, respectively.
        net = layers.Dense(units=16,kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=32,kernel_regularizer=layers.regularizers.l2(1e-6))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Final output layer
        actions = layers.Dense(units=self.action_size,activation='softmax',name='actions')(net)
        self.model = models.Model(inputs=states, outputs=actions)

        # Define the loss function by using action value (Q-value) gradient
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define the optimizer and training function
        optimizer = optimizers.Adam(lr=.00001)
        updates_op = optimizer.get_updates(params = self.model.trainable_weights,loss=loss)
        self.train_fn = K.function(
            inputs = [self.model.input,action_gradients,K.learning_phase()],
            outputs = [],
            updates = updates_op
        )






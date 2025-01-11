"""
This module defines a custom Keras layer, `MultiHeadSelfAttention`, which implements
multi-head self-attention mechanism.

In the context of music genre classification, this layer is applied after the
Convolutional Neural Network (CNN) layers. The CNN extracts spatial features
from the spectrogram segments. The output of the CNN, which can be thought of as
a sequence of feature vectors, is then fed into this attention layer.

Multi-head self-attention allows the model to weigh the importance of different
parts of the feature sequence when making a prediction. It does this by
calculating attention scores between all pairs of feature vectors in the sequence.
These scores determine how much each feature vector should contribute to the
representation of other feature vectors.

The multi-head aspect allows the model to learn different types of relationships
between the feature vectors. Each head focuses on a different subspace of the
input, capturing different aspects of the relationships.

The output of the attention mechanism is then passed through a dropout layer to
prevent overfitting and a layer normalization layer to stabilize training.
The final output is a refined sequence of feature vectors that are more
informative for the classification task.
"""
import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    """
    A custom Keras layer implementing multi-head self-attention.

    This layer applies multi-head self-attention to an input tensor, followed by
    dropout and layer normalization. It's designed to capture relationships within
    the input sequence.

    Args:
        embed_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.1.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by number of heads"
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.head_dim
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization()
        
    def call(self, inputs, training=None):
        """
        Forward pass of the multi-head self-attention layer.

        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool, optional): Whether the layer is in training mode. Defaults to None.

        Returns:
            tf.Tensor: The output tensor after applying multi-head self-attention, dropout, and layer normalization.
        """
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs
        )
        attention_output = self.dropout(attention_output, training=training)
        return self.layernorm(inputs + attention_output) 
    
    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: A dictionary containing the layer's configuration.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout.rate,
        })
        return config
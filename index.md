---
layout: page
---

# Neural Network Building Blocks

Deep learning is hot. New architectures are published all the time, setting the state of the art on a huge variety of different tasks. Once an obscure sub-branch of machine learning, articles on the topic are now popping up on Hacker News almost daily.

Research is outpacing application. New architectures are being created so frequently that it's difficult just to keep track of what's out there. The machine learning practitioner is drowning in options: should I use an RNN or a feedforward architecture? Wait, does that "R" stand for recurrent or recursive? Attentional models seem promising - have they been applied to this problem? I've heard of researchers applying convolutional networks to text processing - when is that appropriate?

Neural network architectures tend to be highly modular. Certain components are used over and over in different architectures. LSTMs, for example have been one of the darlings of 2015, finding application in all kinds of different networks. Much of the recent innovation in deep learning has been in using these components together in new and exciting ways. Connect a convolutional net to an LSTM and you get a video processor.

This article is an attempt to catalog these building blocks and make it easier to understand when and how they should be used.

## Layers, units
Neural networks are made up of layers. Sometimes we construct a single logical layer out of multiple simple layers. These compound layers are sometimes called units, as in "Gated Recurrent Unit". 

A layer takes a vector as input and produces another vector, not necessarily of the same size, as output.

### Vanilla
A simple multiplication of the input vector by a weight matrix, typically passed through a non-linearity.

**Useful for:** mapping from one feature space to another

### Softmax
Takes a vector as input, outputs a vector whose elements are each between 0 and 1 and also sum to 1. This output vector can be interpreted as a categorical probability distribution.

**Useful for:** multiclass classification, attention

### Embedding
A lookup table from some symbol to a vector that represents it in a high dimensional space.

Embedding layers often find use in natural language processing tasks, where they can be used to map from a word to a vector that represents the semantics of that word which can then be used as normal with other layers.

Embedding layers are often represented as matrices where each row is the vector for a particular symbol. These matrices can be multiplied by a one-hot vector for the target symbol to retrieve the embedded vector.

**Useful for:** converting discrete symbols into dense vector representations

### Max Pooling
Returns the maximum value from its input. Can be interpreted as finding the most active of its inputs. Often used in conjunction with convolutional layers.

**Useful for:** dimensionality reduction, especially in image or speech processing

### Gated Recurrent Unit (GRU)
As the name suggests, these units are primarily used in recurrent neural networks. Applied over a sequence, GRUs have internal layers (gates) that allow the network to learn to control how information from earlier in the sequence interacts with information later in the sequence.

**Useful for:** processing sequences with long-distance dependencies

### Long Short-Term Memory (LSTM)
LSTMs are used primarily in recurrent neural networks. They maintain internal state, a vector typically referred to as a memory cell, across applications over a sequence of inputs. They have internal layers (gates) that allow the network to learn to control how information flows in to and out of that memory cell upon processing an element of a sequence. These gates include.

**Useful for:** processing sequences with long-distance dependencies

## Non-linearities
Non-linearities are critical to the expressive power of neural networks - they allow networks to represent non-linear functions. A neural network layer typically applies some linear transformation to its input and then passes the transformed values through an element-wise non-linearity to yield its final output, often called an activation. 

### Sigmoid
A fully differentiable non-linearity that produces activations between 0 and 1.

The sigmoid function and the hyperbolic tangent are related by a simple identity. The general guidance is to use the hyperbolic tangent instead of the signmoid except when one specifically needs activations between 0 and 1. See [jpmuc's answer at Cross Validated](http://stats.stackexchange.com/a/101563) for more details.

**Useful for:** any time you need activations between 0 and 1, for example if the output will be interpreted as a probability

### Hyperbolic Tangent (tanh)
A fully differentiable non-linearity that produces activations between -1 and 1.

**Useful for:** Just about any time you need a non-linearity and the -1 to 1 output range is acceptable for the domain.

### Rectifier
A non-linearity that is simply the max of 0 and the input.

The rectifier is useful for avoiding the vanishing gradient problem. The gradient of a rectifier is either 0 or 1, so as error messages flow through it, they are either passed through unchanged, or zeroed out.

**Useful for:** deep networks, like recurrent networks over long sequences or deep convolution networks

## Higher-order mechanisms
Just about every neural network has more than a single layer. Neural network architectures typically combine individual layers according to recognizable patterns. These higher-order mechanisms give the network the power to operate over different kinds of inputs, perform more complex kinds of computation, or produce different kinds of outputs.

### Stacked layers
The composition (in the sense of function composition) of two layers. The outputs of one layer are simply passed as the inputs to the next.

Stacking is one of the primary ways that neural networks become deep. Some architectures contain dozens of stacked layers. Stacking enables more efficient representations of complex functions (in terms of parameter count) than simply making a layer wider. Later layers in a stack can benefit from the processing that has already been done by earlier layers.

**Useful for:** Increasing a model's expressive power

### Convolution
The application of a single layer to sliding, overlapping windows of an input.

Convolution produces an output that has one more dimension than the convolved layer. If you convolve a layer that produces a scalar output for each window, the result will be a vector of those scalars, one for each window.

Since each application of the convolved layer only sees a (typically small) window of the input, convolution is useful for detecting local features of the input. In an image processing task, these local features might be the edges of objects, for example.

Convolution applies the same layer to eachw indow. The result of each application is invariant to *which* window it's being applied to. This means that it will recognize a local feature equally well no matter where in the input it appears. In the context of object classification, this might mean that the network can recognize an object equally well no matter where in image the object appears - translation invariance.

**Useful for:** Detecting local features, achieving translation invariance

(Tuple vec -> vec) -> ([vec] -> matrix)

### Recurrence
The application of a layer to a sequence of inputs where the result of one application can influence the next.

Recurrence allows for the stateful processing of a sequence of inputs. What all recurrent neural networks have in common is:

- They process sequences of inputs
- The processing of one element of a sequence can influence the processing of the next

The way that the processing of one input can influence the next depends on the layers that are used. The simplest recurrent neural network is a vanilla feed forward layer where the result of the application of the layer to one element of the input sequence is passed as part of the input vector to the same layer when it processes the next element.

Recurrent neural networks have been shown to be Turing complete [citation needed].

**Useful for:** Stateful sequence processing

(vec -> vec) -> ([vec] -> vec)
 
### Recursion
The recursive application of a layer to a tree-structured input.

The input to a recursive neural network is a tree structure of input vectors. We apply a layer that takes a the child vectors of a node in the tree and produces a vector that represents the entire subtree at that node. We apply this layer recursively, starting with the leaf nodes, until we have a vector that represents the entire tree.

**Useful for:** Processing inputs with recursive structure, like sentence parse trees. 

(Tuple vec -> vec) -> (Tree vec -> vec)

## Architectural patterns 
Organize by application domain? NLP, image recognition, audio

### Skip-gram

---
layout: page
---

{% include toc.html %}

Deep learning is hot. New architectures are published all the time, setting the state of the art on a huge variety of different tasks.

Research is outpacing application. Exotic new architectures are being created so frequently that it's difficult to keep track of what's out there. The practitioner is drowning in options.

It helps to take a modular perspective. Each new paper features a new architecture, but usually it's just a new arrangement of familliar building blocks, maybe with one or two totally novel components. 

This page is intended as a reference guide to the building blocks of neural networks. It covers how they work, what they accomplish, and how to use them.

## Layers and units
{: .section-header}

Neural networks are made up of layers. Sometimes we construct a single logical layer out of multiple simple layers. These compound layers are sometimes called units, as in "Gated Recurrent Unit". 

A layer takes a vector as input and produces another vector, not necessarily of the same size, as output.

### Vanilla
{: .section-header}

A linear transformation of the input followed by a nonlinearity.

{% include start-collapse.html id="vanilla-math" class="math" title="Math" %}
A vanilla layer $$h$$ can be defined for a weight matrix $$W$$, bias vector $$b$$, and a nonlinearity $$f$$:

$$h(x) = f(Wx + b)$$

{% include end-collapse.html %}

**Useful for:** mapping from one feature space to another

### Softmax
{: .section-header}

Turns an input vector into a categorical probability distribution.

Takes an input vector and returns another vector of the same dimensionality whose elements are each between 0 and 1 and also sum to 1. This makes the output vector suitable to be interpreted as a categorical probability distribution.

{% include start-collapse.html id="softmax-math" class="math" title="Math" %}
For an $$n$$-dimensional input vector $$x$$, each element $$i$$ of a softmax layer's output is defined as:

$$softmax(x)_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$

{% include end-collapse.html %}

**Useful for:** multiclass classification, attention

### Embedding
{: .section-header}

A lookup table from some symbol to a vector that represents it in a high dimensional space.

Embedding layers often find use in natural language processing tasks, where they can be used to map from a word to a vector that represents the semantics of that word which can then be used as normal with other layers.

Embedding layers are often represented as matrices where each row is the vector for a particular symbol. These matrices can be multiplied by a one-hot vector for the target symbol to retrieve the embedded vector.

**Useful for:** converting discrete symbols into dense vector representations

### Max pooling
{: .section-header}

Downsamples the input, selecting the maximum value in each region.

A max pooling layer segments the input into regions and returns the max value from each. Typically, the result of a pooling operation is a tensor with the same number of axes as the input, but with fewer indices along each axis i.e. the input tensor is shrunk, but not flattened.

**Useful for:** dimensionality reduction, especially in image or speech processing

### Gated Recurrent Unit (GRU)
{: .section-header}

A recurrent unit that uses gates to manage long-term dependencies.

The output vector or hidden state of a GRU at a given time step in a sequence depends on the current element of the sequence as well as the network's hidden state at the previous time step.

The way that previous hidden state influences next hidden state is controlled by two internal layers called gates, the update gate and the reset gate.

{% include start-collapse.html id="gru-math" class="math" title="Math" %}
At each time step, we compute the values of the two gates, the update gate $$z_t$$, and the reset gate $$r_t$$. They are parameterized by $$W^{(z)}$$ and $$W^{(r)}$$, weight matrices for incorporating the current element of the input sequence, $$x_t$$, and $$U^{(z)}$$ and $$U^{(r)}$$, weight matrices for incorporating the previous hidden state, $$h_t$$. We use the sigmoid activation function $$\sigma$$ to ensure that the elements of these gate vectors are between zero and one:

$$
z_t = \sigma(W^{(z)}x_t + U^{(z)}h_{t-1}) \\
r_t = \sigma(W^{(r)}x_t + U^{(r)}h_{t-1})
$$

We then compute $$\widetilde{h}_t$$, often called the proposed hidden state update. $$\circ$$ is the [Hadamard or elementwise product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)). The reset gate vector $$r_t$$ can be seen as controlling how much each element of the previous hidden state is allowed to enter into the proposed hidden state update $$\widetilde{h}_t$$. If an element of $$r_t$$ is close to zero, $$h_{t-1}$$ has little influence on the next hidden state; the state is reset.

$$
\widetilde{h}_t = \tanh(Wx_t + r_t \circ Uh_{t-1})
$$

Finally, we combine the proposed hidden state update with the previous hidden state according to the update gate vector, $$z_t$$. If an element of $$z_t$$ is close to 1, the corresponding element of the final hidden state $$h_t$$ will come almost entirely from $$\widetilde{h}_t$$. If it's close to zero, corresponding element of $$h_{t-1}$$ is carried through almost unchanged.

$$
h_t = z_t \circ h_{t-1} + (1 - z_t) \circ \widetilde{h}_t
$$

{% include end-collapse.html %}

**Useful for:** processing sequences with long-distance dependencies

### Long Short-Term Memory (LSTM)
{: .section-header}

A recurrent unit that uses gates and an internal memory cell to manage long-term dependencies.

LSTMs are used primarily in recurrent neural networks. They maintain internal state, a vector typically referred to as a memory cell, across applications over a sequence of inputs. They have internal layers (gates) that allow the network to learn to control how information flows into and out of that memory cell upon processing an element of a sequence. These gates include.

{% include start-collapse.html id="lstm-math" class="math" title="Math" %}
At each time step, we compute the values of the three gates, the input gate $$i_t$$, the forget gate $$f_t$$, and the output gate $$o_t$$ for an input element $$x_t$$ and the previous hidden state $$h_{t-1}$$. These gates are parameterized by $$W^{(i)}$$, $$W^{(f)}$$, and $$W^{(o)}$$, weight matrices for incorporating the current element of the input sequence, and by $$U^{(i)}$$, $$U^{(f)}$$, and $$U^{(o)}$$, weight matrices for incorporating the previous hidden state:

$$
i_t = \sigma(W^{(i)}x_t + U^{(i)}h_{t-1}) \\
f_t = \sigma(W^{(f)}x_t + U^{(f)}h_{t-1}) \\
o_t = \sigma(W^{(o)}x_t + U^{(o)}h_{t-1})
$$

To compute the new value of the memory cell $$c_t$$, we first compute a proposed new memory cell $$\widetilde{c}_t$$:

$$ \widetilde{c}_t = tanh(W^{(c)}x_t + U^{(c)}h_{t-1}) $$

The final memory cell is computed by combining the proposed new memory cell and the previous memory cell according to the forget and input gates. If $$f_t$$ is close to zero, the previous value of the memory cell is "forgotten". If $$i_t$$ is close to zero, the new value of the memory cell is mostly unaffected by the current input element:

$$ c_t = f_t \circ c_{t-1} + i_t \circ \widetilde{c}_t $$

Finally, we compute the new hidden state $$h_t$$. The output gate $$o_t$$ controls the degree to which the value of the memory cell is output into the LSTM's hidden state:

$$ h_t = o_t \circ tanh(c_t) $$

{% include end-collapse.html %}

**Useful for:** processing sequences with long-distance dependencies

## Nonlinearities
{: .section-header}

Nonlinearities are critical to the expressive power of neural networks - they allow networks to represent nonlinear functions. A neural network layer typically applies some linear transformation to its input and then passes the transformed values through an element-wise nonlinearity to yield its final output, often called an activation. 

### Sigmoid
{: .section-header}

A fully differentiable nonlinearity that produces activations between 0 and 1.

The sigmoid function and the hyperbolic tangent are related by a simple identity. The general guidance is to use the hyperbolic tangent instead of the signmoid except when one specifically needs activations between 0 and 1. See [jpmuc's answer at Cross Validated](http://stats.stackexchange.com/a/101563) for more details.

{% include start-collapse.html id="sigmoid-math" class="math" title="Math" %}
The sigmoid function, often denoted $$\sigma$$, is defined for a scalar $$x$$ as:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

{% include end-collapse.html %}

**Useful for:** any time you need activations between 0 and 1, for example if the output will be interpreted as a probability or as a gate.

### Hyperbolic tangent (tanh)
{: .section-header}

A fully differentiable nonlinearity that produces activations between -1 and 1.

{% include start-collapse.html id="tanh-math" class="math" title="Math" %}
The hyperbolic tangent $$tanh$$ can be defined for a scalar $$x$$ as:

$$ tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1} $$

{% include end-collapse.html %}

**Useful for:** Just about any time you need a nonlinearity and the -1 to 1 output range is acceptable for the domain.

### Rectifier
{: .section-header}

A nonlinearity that is simply the max of 0 and the input.

The rectifier is useful for avoiding the vanishing gradient problem. The gradient of a rectifier is either 0 or 1, so as error messages flow through it, they are either passed through unchanged, or zeroed out.

{% include start-collapse.html id="rectifier-math" class="math" title="Math" %}
The rectifier can be defined for a scalar $$x$$ as:

$$ f(x) = max(0, x) $$

{% include end-collapse.html %}

**Useful for:** deep networks, like recurrent networks over long sequences or deep convolution networks

## Higher-order mechanisms
{: .section-header}

Just about every neural network has more than a single layer. Neural network architectures typically combine individual layers according to recognizable patterns. These higher-order mechanisms give the network the power to operate over different kinds of inputs, perform more complex kinds of computation, or produce different kinds of outputs.

### Stacked layers
{: .section-header}

The composition (in the sense of function composition) of two layers. The outputs of one layer are simply passed as the inputs to the next.

Stacking is one of the primary ways that neural networks become deep. Some architectures contain dozens of stacked layers. Stacking enables more efficient representations of complex functions (in terms of parameter count) than simply making a layer wider. Later layers in a stack can benefit from the processing that has already been done by earlier layers.

**Useful for:** Increasing a model's expressive power

### Convolution
{: .section-header}

The application of a single layer to sliding, overlapping windows of an input.

Convolution produces an output that has one more dimension than the convolved layer. If you convolve a layer that produces a scalar output for each window, the result will be a vector of those scalars, one for each window.

Since each application of the convolved layer only sees a (typically small) window of the input, convolution is useful for detecting local features of the input. In an image processing task, these local features might be the edges of objects, for example.

Convolution applies the same layer to eachw indow. The result of each application is invariant to *which* window it's being applied to. This means that it will recognize a local feature equally well no matter where in the input it appears. In the context of object classification, this might mean that the network can recognize an object equally well no matter where in image the object appears - translation invariance.

**Useful for:** Detecting local features, achieving translation invariance

### Recurrence
{: .section-header}

The application of a layer to a sequence of inputs where the result of one application can influence the next.

Recurrence allows for the stateful processing of a sequence of inputs. What all recurrent neural networks have in common is:

- They process sequences of inputs
- The processing of one element of a sequence can influence the processing of the next

The way that the processing of one input can influence the next depends on the layers that are used. The simplest recurrent neural network is a vanilla feed forward layer where the result of the application of the layer to one element of the input sequence is passed as part of the input vector to the same layer when it processes the next element.

**Useful for:** Stateful sequence processing
 
### Recursion
{: .section-header}

The recursive application of a layer to a tree-structured input.

The input to a recursive neural network is a tree structure of input vectors. We apply a layer that takes a the child vectors of a node in the tree and produces a vector that represents the entire subtree at that node. We apply this layer recursively, starting with the leaf nodes, until we have a vector that represents the entire tree.

**Useful for:** Processing inputs with recursive structure, like sentence parse trees. 

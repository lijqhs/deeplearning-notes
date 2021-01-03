# Course 5: Sequence Models

- [Course 5: Sequence Models](#course-5-sequence-models)
  - [Week 1: Recurrent Neural Networks](#week-1-recurrent-neural-networks)
    - [Recurrent Neural Networks](#recurrent-neural-networks)
      - [Why sequence models](#why-sequence-models)
      - [Notation](#notation)
      - [Recurrent Neural Network Model](#recurrent-neural-network-model)
      - [Backpropagation through time](#backpropagation-through-time)
      - [Different types of RNNs](#different-types-of-rnns)
      - [Language model and sequence generation](#language-model-and-sequence-generation)
      - [Sampling novel sequences](#sampling-novel-sequences)
      - [Vanishing gradients with RNNs](#vanishing-gradients-with-rnns)
      - [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
      - [Long Short Term Memory (LSTM)](#long-short-term-memory-lstm)
      - [Bidirectional RNN](#bidirectional-rnn)
      - [Deep RNNs](#deep-rnns)

## Week 1: Recurrent Neural Networks

>Learn about recurrent neural networks. This type of model has been proven to perform extremely well on temporal data. It has several variants including LSTMs, GRUs and Bidirectional RNNs, which you are going to learn about in this section.

### Recurrent Neural Networks

#### Why sequence models

Examples of sequence data:

- Speech recognition
- Music generation
- Sentiment classification
- DNA sequence analysis
- Machine translation
- Video activity recognition
- Named entity recognition

#### Notation

For a motivation, in the problem of Named Entity Recognition (NER), we have the following notation:

- `x` is the input sentence, such as: `Harry Potter and Hermione Granger invented a new spell.`
- `y` is the output, in this case: `1 1 0 1 1 0 0 0 0`.
- x<sup>\<t></sup> denote the word in the index `t` and y<sup>\<t></sup> is the correspondent output.
- In the *i*th input example, x<sup>(i)\<t></sup> is *t*th word and T<sup>x(i)</sup> is the length of the *i*th example.
- T<sub>y</sub> is the length of the output. In NER, we have T<sub>x</sub> = T<sub>y</sub>.

Words representation introduced in this video is the One-Hot representation.

- First, you have a dictionary which words appear in a certain order.
- Second, for a particular word, we create a new vector with `1` in position of the word in the dictionary and `0` everywhere else.

For a word not in your vocabulary, we need create a new token or a new fake word called unknown word denoted by `<UNK>`.

#### Recurrent Neural Network Model

If we build a neural network to learn the mapping from x to y using the one-hot representation for each word as input, it might not work well. There are two main problems:

- Inputs and outputs can be different lengths in different examples. not every example has the same input length T<sub>x</sub> or the same output length T<sub>y</sub>. Even with a maximum length, zero-padding every input up to the maximum length doesn't seem like a good representation.
- For a naive neural network architecture, it doesn't share features learned across different positions of texts.

*Recurrent Neural Networks*:

- A recurrent neural network does not have either of these disadvantages.
- At each time step, the recurrent neural network that passes on as activation to the next time step for it to use. 
- The recurrent neural network scans through the data from left to right. The parameters it uses for each time step are shared.
- One limitation of unidirectional neural network architecture is that the prediction at a certain time uses inputs or uses information from the inputs earlier in the sequence but not information later in the sequence.
  - `He said, "Teddy Roosevelt was a great president."`
  - `He said, "Teddy bears are on sale!"`
  - You can't tell the difference if you look only at the first three words.

![rnn-forward](img/rnn-forward.png)

Instead of carrying around two parameter matrices W<sub>aa</sub> and W<sub>ax</sub>, we can simplifying the notation by compressing them into just one parameter matrix W<sub>a</sub>.

![rnn-notation](img/rnn-notation.png)

#### Backpropagation through time

In the backpropagation procedure the most significant messaage or the most significant recursive calculation is which goes from right to left, that is, backpropagation through time.

#### Different types of RNNs

There are different types of RNN:

- One to One
- One to Many
- Many to One
- Many to Many

![rnn-type](img/rnn-type.png)

See more details about RNN by [Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

#### Language model and sequence generation

So what a language model does is to tell you what is the probability of a particular sentence.

For example, we have two sentences from speech recognition application:

| sentence | probability |
| :---- | :---- |
| The apple and pair salad. | 𝑃(The apple and pair salad)=3.2x10<sup>-13</sup> |
| The apple and pear salad. | 𝑃(The apple and pear salad)=5.7x10<sup>-10</sup> |

For language model it will be useful to represent a sentence as output `y` rather than inputs `x`. So what the language model does is to estimate the probability of a particular sequence of words `𝑃(y<1>, y<2>, ..., y<T_y>)`.

*How to build a language model*?

`Cats average 15 hours of sleep a day <EOS>` Totally 9 words in this sentence.

- The first thing you would do is to tokenize this sentence.
- Map each of these words to one-hot vectors or indices in vocabulary.
  - Maybe need to add extra token for end of sentence as `<EOS>` or unknown words as `<UNK>`.
  - Omit the period. if you want to treat the period or other punctuation as explicit token, then you can add the period to you vocabulary as well.
- Set the inputs x<sup>\<t></sup> = y<sup>\<t-1></sup>.
- What `a1` does is it will make a softmax prediction to try to figure out what is the probability of the first words y<sup><1></sup>. That is what is the probability of any word in the dictionary. Such as, what's the chance that the first word is *Aaron*?
- Until the end, it will predict the chance of `<EOS>`.
- Define the cost function. The overall loss is just the sum over all time steps of the loss associated with the individual predictions.

![language model](img/lm.png)

If you train this RNN on a large training set, we can do:

- Given an initial set of words, use the model to predict the chance of the next word.
- Given a new sentence `y<1>,y<2>,y<3>`, use it to figure out the chance of this sentence: `p(y<1>,y<2>,y<3>) = p(y<1>) * p(y<2>|y<1>) * p(y<3>|y<1>,y<2>)`

#### Sampling novel sequences

After you train a sequence model, one way you can informally get a sense of what is learned is to have it sample novel sequences.

*How to generate a randomly chosen sentence from your RNN language model*:

- In the first time step, sample what is the first word you want your model to generate: randomly sample according to the softmax distribution.
  - What the softmax distribution gives you is it tells the chance of the first word is 'a', the chance of the first word is 'Aaron', the chance of the first word is 'Zulu', or the chance of the first word refers to `<UNK>` or `<EOS>`. All these probabilities can form a vector.
  - Take the vector and use `np.random.choice` to sample according to distribution defined by this vector probabilities. That lets you sample the first word.
- In the second time step, remember in the last section, y<sup><1></sup> is expected as input. Here take y&#770;<sup><1></sup> you just sampled and pass it as input to the second step. Then use `np.random.choice` to sample y&#770;<sup><2></sup>. Repeat this process until you generate an `<EOS>` token.
- If you want to make sure that your algorithm never generate `<UNK>`, just reject any sample that come out as `<UNK>` and keep resampling from vocabulary until you get a word that's not `<UNK>`.

*Character level language model*:

If you build a character level language model rather than a word level language model, then your sequence y1, y2, y3, would be the individual characters in your training data, rather than the individual words in your training data. Using a character level language model has some pros and cons. As computers gets faster there are more and more applications where people are, at least in some special cases, starting to look at more character level models.

- Advantages:
  - You don't have to worry about `<UNK>`.
- Disadvantages:
  - The main disadvantage of the character level language model is that you end up with much longer sequences.
  - And so character language models are not as good as word level language models at capturing long range dependencies between how the the earlier parts of the sentence also affect the later part of the sentence.
  - More computationally expensive to train.

#### Vanishing gradients with RNNs

- One of the problems with a basic RNN algorithm is that it runs into vanishing gradient problems.
- Language can have very long-term dependencies, for example:
  - The **cat**, which already ate a bunch of food that was delicious ..., **was** full.
  - The **cats**, which already ate a bunch of food that was delicious, and apples, and pears, ..., **were** full.
- The basic RNN we've seen so far is not very good at capturing very long-term dependencies. It's difficult for the output to be strongly influenced by an input that was very early in the sequence.
- When doing backprop, the gradients should not just decrease exponentially, they may also increase exponentially with the number of layers going through.
- Exploding gradients are easier to spot because the parameters just blow up and you might often see NaNs, or not a numbers, meaning results of a numerical overflow in your neural network computation.
  - One solution to that is apply *gradient clipping*: it is bigger than some threshold, re-scale some of your gradient vector so that is not too big.
- Vanishing gradients is much harder to solve and it will be the subject of GRU or LSTM.

#### Gated Recurrent Unit (GRU)

Gate Recurrent Unit is one of the ideas that has enabled RNN to become much better at capturing very long range dependencies and has made RNN much more effective.

A visualization of the RNN unit of the hidden layer of the RNN in terms of a picture:

![rnn-unit](img/rnn-unit.png)

- The GRU unit is going to have a new variable called `c`, which stands for memory cell.
- c&#771;<sup>\<t></sup> is a candidate for replacing c<sup>\<t></sup>.
- For intuition, think of Γ<sub>u</sub> as being either zero or one most of the time. In practice gamma won't be exactly zero or one.
- Because Γ<sub>u</sub> can be so close to zero, can be 0.000001 or even smaller than that, it doesn't suffer from much of a vanishing gradient problem
- Because when Γ<sub>u</sub> is so close to zero this becomes essentially c<sup>\<t></sup> = c<sup>\<t-1></sup> and the value of c<t> is maintained pretty much exactly even across many many time-steps. So this can help significantly with the vanishing gradient problem and therefore allow a neural network to go on even very long range dependencies.
- In the full version of GRU, there is another gate Γ<sub>r</sub>. You can think of `r` as standing for relevance. So this gate Γ<sub>r</sub> tells you how relevant is c<sup>\<t-1></sup> to computing the next candidate for c<sup>\<t></sup>.

![GRU](img/GRU.png)

*Implementation tips*:

- The asterisks are actually element-wise multiplication.
- If you have 100 dimensional or hidden activation value, then c<sup>\<t></sup>, c&#771;<sup>\<t></sup>, Γ<sub>u</sub> would be the same dimension.
  - If Γ<sub>u</sub> is 100 dimensional vector, then it is really a 100 dimensional vector of bits, the value is mostly zero and one.
  - That tells you of this 100 dimensional memory cell which are the bits you want to update. What these element-wise multiplications do is it just element-wise tells the GRU unit which bits to update at every time-step. So you can choose to keep some bits constant while updating other bits.
  - In practice gamma won't be exactly zero or one.

#### Long Short Term Memory (LSTM)

Fancy explanation: [Understanding LSTM Network](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- For the LSTM we will no longer have the case that a<sup>\<t></sup> is equal to c<sup>\<t></sup>.
- And we're not using relevance gate Γ<sub>r</sub>. Instead, LSTM has update, forget and output gates, Γ<sub>u</sub>, Γ<sub>f</sub> and Γ<sub>o</sub> respectively.

![LSTM-units](img/LSTM-units.png)

One cool thing about this you'll notice is that this red line at the top that shows how, so long as you set the forget and the update gate appropriately, it is relatively easy for the LSTM to have some value c<sup>\<0></sup> and have that be passed all the way to the right to have your, maybe, c<sup>\<3></sup> equals c<sup>\<0></sup>. And this is why the LSTM, as well as the GRU, is very good at memorizing certain values even for a long time, for certain real values stored in the memory cell even for many, many timesteps.

![LSTM](img/LSTM.png)

*One common variation of LSTM*:

- Peephole connection: instead of just having the gate values be dependent only on a<sup>\<t-1></sup>, x<sup>\<t></sup>, sometimes, people also sneak in there the values c<sup>\<t-1></sup> as well.

*GRU vs. LSTM*:

- The advantage of the GRU is that it's a simpler model and so it is actually easier to build a much bigger network, it only has two gates,
so computationally, it runs a bit faster. So, it scales the building somewhat bigger models.
- The LSTM is more powerful and more effective since it has three gates instead of two. If you want to pick one to use, LSTM has been the historically more proven choice. Most people today will still use the LSTM as the default first thing to try.

**Implementation tips**:

- *forget gate Γ<sub>f</sub>*
  * The forget gate Γ<sub>f</sub><sup>\<t></sup> has the same dimensions as the previous cell state c<sup>\<t-1></sup>.
  * This means that the two can be multiplied together, element-wise.
  * Multiplying the tensors Γ<sub>f</sub><sup>\<t></sup> is like applying a mask over the previous cell state.
  * If a single value in Γ<sub>f</sub><sup>\<t></sup> is 0 or close to 0, then the product is close to 0.
    * This keeps the information stored in the corresponding unit in c<sup>\<t-1></sup> from being remembered for the next time step.
  * Similarly, if one value is close to 1, the product is close to the original value in the previous cell state.
    * The LSTM will keep the information from the corresponding unit of c<sup>\<t-1></sup>, to be used in the next time step.

- *candidate value c&#771;<sup>\<t></sup>*
  * The candidate value is a tensor containing information from the current time step that **may** be stored in the current cell state c<sup>\<t></sup>.
  * Which parts of the candidate value get passed on depends on the update gate.
  * The candidate value is a tensor containing values that range from -1 to 1. (tanh function)
  * The tilde "~" is used to differentiate the candidate c&#771;<sup>\<t></sup> from the cell state c<sup>\<t></sup>.

- *update gate Γ<sub>u</sub>*
  * The update gate decides what parts of a "candidate" tensor c&#771;<sup>\<t></sup> are passed onto the cell state c<sup>\<t></sup>.
  * The update gate is a tensor containing values between 0 and 1.
    * When a unit in the update gate is close to 1, it allows the value of the candidate c&#771;<sup>\<t></sup> to be passed onto the hidden state c<sup>\<t></sup>.
    * When a unit in the update gate is close to 0, it prevents the corresponding value in the candidate from being passed onto the hidden state.

- *cell state c<sup>\<t></sup>*
  * The cell state is the "memory" that gets passed onto future time steps.
  * The new cell state c<sup>\<t></sup> is a combination of the previous cell state and the candidate value.

- *output gate Γ<sub>o</sub>*
  * The output gate decides what gets sent as the prediction (output) of the time step.
  * The output gate is like the other gates. It contains values that range from 0 to 1.

- *hidden state a<sup>\<t></sup>*
  * The hidden state gets passed to the LSTM cell's next time step.
  * It is used to determine the three gates (Γ<sub>f</sub>, Γ<sub>u</sub>, Γ<sub>o</sub>) of the next time step.
  * The hidden state is also used for the prediction y<sup>\<t></sup>.

#### Bidirectional RNN

![RNN-ner](img/BRNN-ner.png)

- Bidirectional RNN lets you at a point in time to take information from both earlier and later in the sequence.
- This network defines a Acyclic graph
- The forward prop has part of the computation going from left to right and part of computation going from right to left in this diagram.
- So information from x<sup>\<1></sup>, x<sup>\<2></sup>, x<sup>\<3></sup> are all taken into account with information from x<sup>\<4></sup> can flow through a backward four to a backward three to Y three. So this allows the prediction at time three to take as input both information from the past, as well as information from the present which goes into both the forward and the backward things at this step, as well as information from the future.
- Blocks can be not just the standard RNN block but they can also be GRU blocks or LSTM blocks. In fact, BRNN with LSTM units is commonly used in NLP problems.

![BRNN](img/BRNN.png)

*Disadvantage*:

The disadvantage of the bidirectional RNN is that you do need the entire sequence of data before you can make predictions anywhere. So, for example, if you're building a speech recognition system, then the BRNN will let you take into account the entire speech utterance but if you use this straightforward implementation, you need to wait for the person to stop talking to get the entire utterance before you can actually process it and make a speech recognition prediction. For a real type speech recognition applications, they're somewhat more complex modules as well rather than just using the standard bidirectional RNN as you've seen here.

#### Deep RNNs

- For learning very complex functions sometimes is useful to stack multiple layers of RNNs together to build even deeper versions of these models.
- The blocks don't just have to be standard RNN, the simple RNN model. They can also be GRU blocks LSTM blocks.
- And you can also build deep versions of the bidirectional RNN.

![DRNN](img/DRNN.png)

---
Notes by [lijqhs](mailto:azurciel@outlook.com) © 2020


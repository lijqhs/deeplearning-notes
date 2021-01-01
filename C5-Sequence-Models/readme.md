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
- Exploding gradients


#### Gated Recurrent Unit (GRU)

#### Long Short Term Memory (LSTM)

#### Bidirectional RNN

#### Deep RNNs


[course-summary]: https://www.slideshare.net/TessFerrandez/notes-from-coursera-deep-learning-courses-by-andrew-ng


---
Notes by [Aaron](mailto:lijqhs@gmail.com) © 2020

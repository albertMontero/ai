# Natural Language Processing and Large Language Models

https://huggingface.co/learn/llm-course/chapter1/2

Before jumping into Transformer models, let's do a quick overview of what natural language processing is, how large language models have transformed the field, and why we care about it.

## What is NLP?

<Youtube id="iNzlxWUAjd4" />

NLP is a field of linguistics and machine learning focused on understanding everything related to human language. The aim of NLP tasks is not only to understand single words individually, but to be able to understand the context of those words.

The following is a list of common NLP tasks, with some examples of each:

- **Classifying whole sentences**: Getting the sentiment of a review, detecting if an email is spam, determining if a sentence is grammatically correct or whether two sentences are logically related or not
- **Classifying each word in a sentence**: Identifying the grammatical components of a sentence (noun, verb, adjective), or the named entities (person, location, organization)
- **Generating text content**: Completing a prompt with auto-generated text, filling in the blanks in a text with masked words
- **Extracting an answer from a text**: Given a question and a context, extracting the answer to the question based on the information provided in the context
- **Generating a new sentence from an input text**: Translating a text into another language, summarizing a text

NLP isn't limited to written text though. It also tackles complex challenges in speech recognition and computer vision, such as generating a transcript of an audio sample or a description of an image.

## The Rise of Large Language Models (LLMs)

In recent years, the field of NLP has been revolutionized by Large Language Models (LLMs). These models, which include architectures like GPT (Generative Pre-trained Transformer) and [Llama](https://huggingface.co/meta-llama), have transformed what's possible in language processing.

A large language model (LLM) is an AI model trained on massive amounts of text data that can understand and generate human-like text, recognize patterns in language, and **perform a wide variety of language tasks without task-specific training**. They represent a significant advancement in the field of natural language processing (NLP).

LLMs are characterized by:

- **Scale**: They contain millions, billions, or even hundreds of billions of parameters
- **General capabilities**: They can perform multiple tasks without task-specific training
- **In-context learning**: They can learn from examples provided in the prompt
- **Emergent abilities**: As these models grow in size, they demonstrate capabilities that weren't explicitly programmed or anticipated

The advent of LLMs **has shifted the paradigm from building specialized models for specific NLP tasks to using a single, large model that can be prompted or fine-tuned to address a wide range of language tasks**. This has made sophisticated language processing more accessible while also introducing new challenges in areas like efficiency, ethics, and deployment.

However, LLMs also have important limitations:

- **Hallucinations**: They can generate incorrect information confidently
- **Lack of true understanding**: They lack true understanding of the world and operate purely on statistical patterns
- **Bias**: They may reproduce biases present in their training data or inputs.
- **Context windows**: They have limited context windows (though this is improving)
- **Computational resources**: They require significant computational resources

## Why is language processing challenging?

Computers don't process information in the same way as humans. For example, when we read the sentence "I am hungry," we can easily understand its meaning. Similarly, given two sentences such as "I am hungry" and "I am sad," we're able to easily determine how similar they are. For machine learning (ML) models, such tasks are more difficult. The text needs to be processed in a way that enables the model to learn from it. And because language is complex, we need to think carefully about how this processing must be done. There has been a lot of research done on how to represent text, and we will look at some methods in the next chapter.

Even with the advances in LLMs, many fundamental challenges remain. These include understanding ambiguity, cultural context, sarcasm, and humor. LLMs address these challenges through massive training on diverse datasets, but still often fall short of human-level understanding in many complex scenarios.

# Transformers, what can they do?

https://huggingface.co/learn/llm-course/chapter1/3

In this section, we will look at what Transformer models can do and use our first tool from the 🤗 Transformers library: the `pipeline()` function.

👀 See that <em>Open in Colab</em> button on the top right? Click on it to open a Google Colab notebook with all the code samples of this section. This button will be present in any section containing code examples.

If you want to run the examples locally, we recommend taking a look at the <a href="/course/chapter0">setup</a>.

## Transformers are everywhere!

Transformer models are used to solve all kinds of tasks across different **modalities**, including natural language processing (NLP), computer vision, audio processing, and more. Here are some of the companies and organizations using Hugging Face and Transformer models, who also contribute back to the community by sharing their models:

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/companies.PNG" alt="Companies using Hugging Face" width="100%">

The [🤗 Transformers library](https://github.com/huggingface/transformers) provides the functionality to create and use those shared models. The [Model Hub](https://huggingface.co/models) contains millions of pretrained models that anyone can download and use. You can also upload your own models to the Hub!

⚠️ The Hugging Face Hub is not limited to Transformer models. Anyone can share any kind of models or datasets they want! <a href="https://huggingface.co/join">Create a huggingface.co</a> account to benefit from all available features!

Before diving into how Transformer models work under the hood, let's look at a few examples of how they can be used to solve some interesting NLP problems.

## Working with pipelines

<Youtube id="tiZFewofSLM" />

The most basic object in the 🤗 Transformers library is the `pipeline()` function. It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

```python out
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```

We can even pass several sentences!

```python
classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
```

```python out
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

By default, this pipeline selects a particular pretrained model that has been fine-tuned for sentiment analysis in English. The model is downloaded and cached when you create the `classifier` object. If you rerun the command, the cached model will be used instead and there is no need to download the model again.

There are three main steps involved when you pass some text to a pipeline:

1. The text is preprocessed into a format the model can understand.
2. The preprocessed inputs are passed to the model.
3. The predictions of the model are post-processed, so you can make sense of them.

## Available pipelines for different modalities

The `pipeline()` function supports multiple modalities, allowing you to work with text, images, audio, and even multimodal tasks. In this course we'll focus on text tasks, but it's useful to understand the transformer architecture's potential, so we'll briefly outline it.

Here's an overview of what's available:

For a full and updated list of pipelines, see the [🤗 Transformers documentation](https://huggingface.co/docs/hub/en/models-tasks).

### Text pipelines

- `text-generation`: Generate text from a prompt
- `text-classification`: Classify text into predefined categories
- `summarization`: Create a shorter version of a text while preserving key information
- `translation`: Translate text from one language to another
- `zero-shot-classification`: Classify text without prior training on specific labels
- `feature-extraction`: Extract vector representations of text

### Image pipelines

- `image-to-text`: Generate text descriptions of images
- `image-classification`: Identify objects in an image
- `object-detection`: Locate and identify objects in images

### Audio pipelines

- `automatic-speech-recognition`: Convert speech to text
- `audio-classification`: Classify audio into categories
- `text-to-speech`: Convert text to spoken audio

### Multimodal pipelines

- `image-text-to-text`: Respond to an image based on a text prompt

Let's explore some of these pipelines in more detail!

## Zero-shot classification

We'll start by tackling a more challenging task where we need to classify texts that haven't been labelled. This is a common scenario in real-world projects because annotating text is usually time-consuming and requires domain expertise. For this use case, the `zero-shot-classification` pipeline is very powerful: it allows you to specify which labels to use for the classification, so you don't have to rely on the labels of the pretrained model. You've already seen how the model can classify a sentence as positive or negative using those two labels — but it can also classify the text using any other set of labels you like.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

```python out
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
```

This pipeline is called _zero-shot_ because you don't need to fine-tune the model on your data to use it. It can directly return probability scores for any list of labels you want!

✏️ **Try it out!** Play around with your own sequences and labels and see how the model behaves.

## Text generation

Now let's see how to use a pipeline to generate some text. The main idea here is that you provide a prompt and the model will auto-complete it by generating the remaining text. This is similar to the predictive text feature that is found on many phones. Text generation involves randomness, so it's normal if you don't get the same results as shown below.

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

```python out
[{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}]
```

You can control how many different sequences are generated with the argument `num_return_sequences` and the total length of the output text with the argument `max_length`.

✏️ **Try it out!** Use the `num_return_sequences` and `max_length` arguments to generate two sentences of 15 words each.

## Using any model from the Hub in a pipeline

The previous examples used the default model for the task at hand, but you can also choose a particular model from the Hub to use in a pipeline for a specific task — say, text generation. Go to the [Model Hub](https://huggingface.co/models) and click on the corresponding tag on the left to display only the supported models for that task. You should get to a page like [this one](https://huggingface.co/models?pipeline_tag=text-generation).

Let's try the [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) model! Here's how to load it in the same pipeline as before:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

```python out
[{'generated_text': 'In this course, we will teach you how to manipulate the world and '
                    'move your mental and physical capabilities to your advantage.'},
 {'generated_text': 'In this course, we will teach you how to become an expert and '
                    'practice realtime, and with a hands on experience on both real '
                    'time and real'}]
```

You can refine your search for a model by clicking on the language tags, and pick a model that will generate text in another language. The Model Hub even contains checkpoints for multilingual models that support several languages.

Once you select a model by clicking on it, you'll see that there is a widget enabling you to try it directly online. This way you can quickly test the model's capabilities before downloading it.

✏️ **Try it out!** Use the filters to find a text generation model for another language. Feel free to play with the widget and use it in a pipeline!

### Inference Providers

All the models can be tested directly through your browser using the Inference Providers, which is available on the Hugging Face [website](https://huggingface.co/docs/inference-providers/en/index). You can play with the model directly on this page by inputting custom text and watching the model process the input data.

Inference Providers that powers the widget is also available as a paid product, which comes in handy if you need it for your workflows. See the [pricing page](https://huggingface.co/docs/inference-providers/en/pricing) for more details.

## Mask filling

The next pipeline you'll try is `fill-mask`. The idea of this task is to fill in the blanks in a given text:

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

```python out
[{'sequence': 'This course will teach you all about mathematical models.',
  'score': 0.19619831442832947,
  'token': 30412,
  'token_str': ' mathematical'},
 {'sequence': 'This course will teach you all about computational models.',
  'score': 0.04052725434303284,
  'token': 38163,
  'token_str': ' computational'}]
```

The `top_k` argument controls how many possibilities you want to be displayed. Note that here the model fills in the special `<mask>` word, which is often referred to as a _mask token_. Other mask-filling models might have different mask tokens, so it's always good to verify the proper mask word when exploring other models. One way to check it is by looking at the mask word used in the widget.

✏️ **Try it out!** Search for the `bert-base-cased` model on the Hub and identify its mask word in the Inference API widget. What does this model predict for the sentence in our `pipeline` example above?

## Named entity recognition

Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to entities such as persons, locations, or organizations. Let's look at an example:

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45},
 {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
]
```

Here the model correctly identified that Sylvain is a person (PER), Hugging Face an organization (ORG), and Brooklyn a location (LOC).

We pass the option `grouped_entities=True` in the pipeline creation function to tell the pipeline to regroup together the parts of the sentence that correspond to the same entity: here the model correctly grouped "Hugging" and "Face" as a single organization, even though the name consists of multiple words. In fact, as we will see in the next chapter, the preprocessing even splits some words into smaller parts. For instance, `Sylvain` is split into four pieces: `S`, `##yl`, `##va`, and `##in`. In the post-processing step, the pipeline successfully regrouped those pieces.

✏️ **Try it out!** Search the Model Hub for a model able to do part-of-speech tagging (usually abbreviated as POS) in English. What does this model predict for the sentence in the example above?

## Question answering

The `question-answering` pipeline answers questions using information from a given context:

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
```

```python out
{'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```

Note that this pipeline works by extracting information from the provided context; it does not generate the answer.

## Summarization

Summarization is the task of reducing a text into a shorter text while keeping all (or most) of the important aspects referenced in the text. Here's an example:

```python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"""
)
```

```python out
[{'summary_text': ' America has changed dramatically during recent years . The '
                  'number of engineering graduates in the U.S. has declined in '
                  'traditional engineering disciplines such as mechanical, civil '
                  ', electrical, chemical, and aeronautical engineering . Rapidly '
                  'developing economies such as China and India, as well as other '
                  'industrial countries in Europe and Asia, continue to encourage '
                  'and advance engineering .'}]
```

Like with text generation, you can specify a `max_length` or a `min_length` for the result.

## Translation

For translation, you can use a default model if you provide a language pair in the task name (such as `"translation_en_to_fr"`), but the easiest way is to pick the model you want to use on the [Model Hub](https://huggingface.co/models). Here we'll try translating from French to English:

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```

```python out
[{'translation_text': 'This course is produced by Hugging Face.'}]
```

Like with text generation and summarization, you can specify a `max_length` or a `min_length` for the result.

✏️ **Try it out!** Search for translation models in other languages and try to translate the previous sentence into a few different languages.

## Image and audio pipelines

Beyond text, Transformer models can also work with images and audio. Here are a few examples:

### Image classification

```python
from transformers import pipeline

image_classifier = pipeline(
    task="image-classification", model="google/vit-base-patch16-224"
)
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(result)
```

```python out
[{'label': 'lynx, catamount', 'score': 0.43350091576576233},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Egyptian cat', 'score': 0.02394474856555462},
 {'label': 'tiger cat', 'score': 0.02288915030658245}]
```

### Automatic speech recognition

```python
from transformers import pipeline

transcriber = pipeline(
    task="automatic-speech-recognition", model="openai/whisper-large-v3"
)
result = transcriber(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
)
print(result)
```

```python out
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## Combining data from multiple sources

One powerful application of Transformer models is their ability to combine and process data from multiple sources. This is especially useful when you need to:

1. Search across multiple databases or repositories
2. Consolidate information from different formats (text, images, audio)
3. Create a unified view of related information

For example, you could build a system that:

- Searches for information across databases in multiple modalities like text and image.
- Combines results from different sources into a single coherent response. For example, from an audio file and text description.
- Presents the most relevant information from a database of documents and metadata.

## Conclusion

The pipelines shown in this chapter are mostly for demonstrative purposes. They were programmed for specific tasks and cannot perform variations of them. In the next chapter, you'll learn what's inside a `pipeline()` function and how to customize its behavior.

# How do Transformers work?

https://huggingface.co/learn/llm-course/chapter1/4

In this section, we will take a look at the architecture of Transformer models and dive deeper into the concepts of attention, encoder-decoder architecture, and more.

🚀 We're taking things up a notch here. This section is detailed and technical, so don't worry if you don't understand everything right away. We'll come back to these concepts later in the course.

## A bit of Transformer history

Here are some reference points in the (short) history of Transformer models:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono.svg" alt="A brief chronology of Transformers models.">

The [Transformer architecture](https://arxiv.org/abs/1706.03762) was introduced in June 2017. The focus of the original research was on translation tasks. This was followed by the introduction of several influential models, including:

- **June 2018**: [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results

- **October 2018**: [BERT](https://arxiv.org/abs/1810.04805), another large pretrained model, this one designed to produce better summaries of sentences (more on this in the next chapter!)

- **February 2019**: [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns

- **October 2019**: [T5](https://huggingface.co/papers/1910.10683), A multi-task focused implementation of the sequence-to-sequence Transformer architecture.

- **May 2020**, [GPT-3](https://huggingface.co/papers/2005.14165), an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called _zero-shot learning_)

- **January 2022**: [InstructGPT](https://huggingface.co/papers/2203.02155), a version of GPT-3 that was trained to follow instructions better
  This list is far from comprehensive, and is just meant to highlight a few of the different kinds of Transformer models. Broadly, they can be grouped into three categories:

- **January 2023**: [Llama](https://huggingface.co/papers/2302.13971), a large language model that is able to generate text in a variety of languages.

- **March 2023**: [Mistral](https://huggingface.co/papers/2310.06825), a 7-billion-parameter language model that outperforms Llama 2 13B across all evaluated benchmarks, leveraging grouped-query attention for faster inference and sliding window attention to handle sequences of arbitrary length.

- **May 2024**: [Gemma 2](https://huggingface.co/papers/2408.00118), a family of lightweight, state-of-the-art open models ranging from 2B to 27B parameters that incorporate interleaved local-global attentions and group-query attention, with smaller models trained using knowledge distillation to deliver performance competitive with models 2-3 times larger.

- **November 2024**: [SmolLM2](https://huggingface.co/papers/2502.02737), a state-of-the-art small language model (135 million to 1.7 billion parameters) that achieves impressive performance despite its compact size, and unlocking new possibilities for mobile and edge devices.

- GPT-like (also called _auto-regressive_ Transformer models)
- BERT-like (also called _auto-encoding_ Transformer models)
- T5-like (also called _sequence-to-sequence_ Transformer models)

We will dive into these families in more depth later on.

## Transformers are language models

All the Transformer models mentioned above (GPT, BERT, T5, etc.) have been trained as _language models_. This means **they have been trained on large amounts of raw text in a self-supervised fashion**.

**Self-supervised learning** is a type of training in which the objective is automatically computed from the inputs of the model. That means that humans are not needed to label the data!

This type of model develops a statistical understanding of the language it has been trained on, but it's **less useful for specific practical tasks**. Because of this, the general pretrained model then goes through a process called **_transfer learning_ or _fine-tuning_**. During this process, the model is fine-tuned in a supervised way -- that is, using human-annotated labels -- on a given task.

An example of a task is predicting the next word in a sentence having read the _n_ previous words. This is called _causal language modeling_ because the output depends on the past and present inputs, but not the future ones.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling.svg" alt="Example of causal language modeling in which the next word from a sentence is predicted.">

Another example is _masked language modeling_, in which the model predicts a masked word in the sentence.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling.svg" alt="Example of masked language modeling in which a masked word from a sentence is predicted.">

## Transformers are big models

Apart from a few outliers (like DistilBERT), the general strategy to achieve better performance is by increasing the models' sizes as well as the amount of data they are pretrained on.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/model_parameters.png" alt="Number of parameters of recent Transformers models" width="90%">
</div>

Unfortunately, training a model, especially a large one, **requires a large amount of data**. This becomes very costly in terms of time and compute resources. It even translates to environmental impact, as can be seen in the following graph.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/carbon_footprint.svg" alt="The carbon footprint of a large language model.">
</div>

<Youtube id="ftWlj4FBHTg"/>

And this is showing a project for a (very big) model led by a team consciously trying to reduce the environmental impact of pretraining. The footprint of running lots of trials to get the best hyperparameters would be even higher.

Imagine if each time a research team, a student organization, or a company wanted to train a model, it did so from scratch. This would lead to huge, unnecessary global costs!

This is why sharing language models is paramount: sharing the trained weights and building on top of already trained weights reduces the overall compute cost and carbon footprint of the community.

By the way, you can evaluate the carbon footprint of your models' training through several tools. For example [ML CO2 Impact](https://mlco2.github.io/impact/) or [Code Carbon](https://codecarbon.io/) which is integrated in 🤗 Transformers. To learn more about this, you can read this [blog post](https://huggingface.co/blog/carbon-emissions-on-the-hub) which will show you how to generate an `emissions.csv` file with an estimate of the footprint of your training, as well as the [documentation](https://huggingface.co/docs/hub/model-cards-co2) of 🤗 Transformers addressing this topic.

## Transfer Learning

<Youtube id="BqqfQnyjmgg" />

_Pretraining_ is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining.svg" alt="The pretraining of a language model is costly in both time and money.">
</div>

This pretraining is usually done on very large amounts of data. Therefore, it requires a very large corpus of data, and training can take up to **several weeks**.

_Fine-tuning_, on the other hand, is the training done **after** a model has been pretrained. To perform fine-tuning, you first acquire a pretrained language model, then perform additional training with a dataset specific to your task. Wait -- why not simply train the model for your final use case from the start (**scratch**)? There are a couple of reasons:

- The pretrained model was already trained on a dataset that has some similarities with the fine-tuning dataset. The **fine-tuning process is thus able to take advantage of knowledge acquired by the initial model during pretraining** (for instance, with NLP problems, the pretrained model will have some kind of statistical understanding of the language you are using for your task).
- Since the pretrained model was already trained on lots of data, the **fine-tuning requires way less data** to get decent results.
- For the same reason, the **amount of time and resources needed to get good results are much lower**.

For example, one could leverage a pretrained model trained on the English language and then fine-tune it on an arXiv corpus, resulting in a science/research-based model. The fine-tuning will only require a limited amount of data: the knowledge the pretrained model has acquired is "transferred," hence the term _transfer learning_.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning.svg" alt="The fine-tuning of a language model is cheaper than pretraining in both time and money.">
</div>

Fine-tuning a model therefore has lower time, data, financial, and environmental costs. It is also quicker and easier to iterate over different fine-tuning schemes, as the training is less constraining than a full pretraining.

This process will also achieve better results than training from scratch (unless you have lots of data), which is why you should always try to leverage a pretrained model -- one as close as possible to the task you have at hand -- and fine-tune it.

## General Transformer architecture

In this section, we'll go over the general architecture of the Transformer model. Don't worry if you don't understand some of the concepts; there are detailed sections later covering each of the components.

<Youtube id="H39Z_720T5s" />

The model is primarily composed of two blocks:

- **Encoder (left)**: The encoder receives an input and builds a representation of it (its features). This means that the model is **optimized to acquire understanding from the input**.
- **Decoder (right)**: The decoder uses the encoder's representation (features) along with other inputs to generate a target sequence. This means that the model is **optimized for generating outputs**.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks.svg" alt="Architecture of a Transformers models">
</div>

Each of these parts can be used independently, depending on the task:

- **Encoder-only models**: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
- **Decoder-only models**: Good for generative tasks such as text generation.
- **Encoder-decoder models** or **sequence-to-sequence models**: Good for generative tasks that require an input, such as translation or summarization.

We will dive into those architectures independently in later sections.

## Attention layers

A key feature of Transformer models is that they are built with special layers called _attention layers_. In fact, the title of the paper introducing the Transformer architecture was ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)! We will explore the details of attention layers later in the course; for now, all you need to know is that **this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others)** when dealing with the representation of each word.

To put this into context, consider the task of translating text from English to French. Given the input "You like this course", a translation model will need to also attend to the adjacent word "You" to get the proper translation for the word "like", because in French the verb "like" is conjugated differently depending on the subject. The rest of the sentence, however, is not useful for the translation of that word. In the same vein, when translating "this" the model will also need to pay attention to the word "course", because "this" translates differently depending on whether the associated noun is masculine or feminine. Again, the other words in the sentence will not matter for the translation of "course". With more complex sentences (and more complex grammar rules), the model would need to pay special attention to words that might appear farther away in the sentence to properly translate each word.

The same concept applies to any task associated with natural language: a word by itself has a meaning, but that meaning is deeply affected by the context, which can be any other word (or words) before or after the word being studied.

Now that you have an idea of what attention layers are all about, let's take a closer look at the Transformer architecture.

## The original architecture

The Transformer architecture was originally designed for translation. During training, the encoder receives inputs (sentences) in a certain language, while the decoder receives the same sentences in the desired target language. In the encoder, the attention layers can use all the words in a sentence (since, as we just saw, the translation of a given word can be dependent on what is after as well as before it in the sentence). The decoder, however, works sequentially and can only pay attention to the words in the sentence that it has already translated (so, only the words before the word currently being generated). For example, when we have predicted the first three words of the translated target, we give them to the decoder which then uses all the inputs of the encoder to try to predict the fourth word.

To speed things up during training (when the model has access to target sentences), the decoder is fed the whole target, but it is not allowed to use future words (if it had access to the word at position 2 when trying to predict the word at position 2, the problem would not be very hard!). For instance, when trying to predict the fourth word, the attention layer will only have access to the words in positions 1 to 3.

The original Transformer architecture looked like this, with the encoder on the left and the decoder on the right:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg" alt="Architecture of a Transformers models">
</div>

Note that the first attention layer in a decoder block pays attention to all (past) inputs to the decoder, but the second attention layer uses the output of the encoder. It can thus access the whole input sentence to best predict the current word. This is very useful as different languages can have grammatical rules that put the words in different orders, or some context provided later in the sentence may be helpful to determine the best translation of a given word.

The _attention mask_ can also be used in the encoder/decoder to prevent the model from paying attention to some special words -- for instance, the special padding word used to make all the inputs the same length when batching together sentences.

## Architectures vs. checkpoints[[architecture-vs-checkpoints]]

As we dive into Transformer models in this course, you'll see mentions of _architectures_ and _checkpoints_ as well as _models_. These terms all have slightly different meanings:

- **Architecture**: This is the skeleton of the model -- the definition of each layer and each operation that happens within the model.
- **Checkpoints**: These are the weights that will be loaded in a given architecture.
- **Model**: This is an umbrella term that isn't as precise as "architecture" or "checkpoint": it can mean both. This course will specify _architecture_ or _checkpoint_ when it matters to reduce ambiguity.

For example, BERT is an architecture while `bert-base-cased`, a set of weights trained by the Google team for the first release of BERT, is a checkpoint. However, one can say "the BERT model" and "the `bert-base-cased` model."

# Transformer Architectures

In the previous sections, we introduced the general Transformer architecture and explored how these models can solve various tasks. Now, let's take a closer look at the three main architectural variants of Transformer models and understand when to use each one. Then, we looked at how those architectures are applied to different language tasks.

In this section, we're going to dive deeper into the three main architectural variants of Transformer models and understand when to use each one.

Remember that most Transformer models use one of three architectures: encoder-only, decoder-only, or encoder-decoder (sequence-to-sequence). Understanding these differences will help you choose the right model for your specific task.

## Encoder models

<Youtube id="MUqNwgPjJvQ" />

Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having "bi-directional" attention, and are often called _auto-encoding models_.

The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.

Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence classification, named entity recognition (and more generally word classification), and extractive question answering.

As we saw in [How 🤗 Transformers solve tasks](https://huggingface.co/learn/llm-course/chapter1/5), encoder models like BERT excel at understanding text because they can look at the entire context in both directions. This makes them perfect for tasks where comprehension of the whole input is important.

Representatives of this family of models include:

- [BERT](https://huggingface.co/docs/transformers/model_doc/bert)
- [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [ModernBERT](https://huggingface.co/docs/transformers/en/model_doc/modernbert)

## Decoder models

<Youtube id="d_ixlCubqQw" />

Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called _auto-regressive models_.

The pretraining of decoder models usually revolves around predicting the next word in the sentence.

These models are best suited for tasks involving text generation.

Decoder models like GPT are designed to generate text by predicting one token at a time. As we explored in [How 🤗 Transformers solve tasks](https://huggingface.co/learn/llm-course/chapter1/5), they can only see previous tokens, which makes them excellent for creative text generation but less ideal for tasks requiring bidirectional understanding.

Representatives of this family of models include:

- [Hugging Face SmolLM Series](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Meta's Llama Series](https://huggingface.co/docs/transformers/en/model_doc/llama4)
- [Google's Gemma Series](https://huggingface.co/docs/transformers/main/en/model_doc/gemma3)
- [DeepSeek's V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)

### Modern Large Language Models (LLMs)

Most modern Large Language Models (LLMs) use the decoder-only architecture. These models have grown dramatically in size and capabilities over the past few years, with some of the largest models containing hundreds of billions of parameters.

Modern LLMs are typically trained in two phases:

1. **Pretraining**: The model learns to predict the next token on vast amounts of text data
2. **Instruction tuning**: The model is fine-tuned to follow instructions and generate helpful responses

This approach has led to models that can understand and generate human-like text across a wide range of topics and tasks.

#### Key capabilities of modern LLMs

Modern decoder-based LLMs have demonstrated impressive capabilities:

| Capability         | Description                                      | Example                                         |
| ------------------ | ------------------------------------------------ | ----------------------------------------------- |
| Text generation    | Creating coherent and contextually relevant text | Writing essays, stories, or emails              |
| Summarization      | Condensing long documents into shorter versions  | Creating executive summaries of reports         |
| Translation        | Converting text between languages                | Translating English to Spanish                  |
| Question answering | Providing answers to factual questions           | "What is the capital of France?"                |
| Code generation    | Writing or completing code snippets              | Creating a function based on a description      |
| Reasoning          | Working through problems step by step            | Solving math problems or logical puzzles        |
| Few-shot learning  | Learning from a few examples in the prompt       | Classifying text after seeing just 2-3 examples |

You can experiment with decoder-based LLMs directly in your browser via model repo pages on the Hub. Here's an an example with the classic [GPT-2](https://huggingface.co/openai-community/gpt2) (OpenAI's finest open source model!):

## Sequence-to-sequence models

<Youtube id="0_4KEb08xrE" />

Encoder-decoder models (also called _sequence-to-sequence models_) use both parts of the Transformer architecture. At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input.

The pretraining of these models can take different forms, but it often involves reconstructing a sentence for which the input has been somehow corrupted (for instance by masking random words). The pretraining of the T5 model consists of replacing random spans of text (that can contain several words) with a single mask special token, and the task is then to predict the text that this mask token replaces.

Sequence-to-sequence models are best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation, or generative question answering.

As we saw in [How 🤗 Transformers solve tasks](https://huggingface.co/learn/llm-course/chapter1/5), encoder-decoder models like BART and T5 combine the strengths of both architectures. The encoder provides deep bidirectional understanding of the input, while the decoder generates appropriate output text. This makes them perfect for tasks that transform one sequence into another, like translation or summarization.

### Practical applications

Sequence-to-sequence models excel at tasks that require transforming one form of text into another while preserving meaning. Some practical applications include:

| Application             | Description                                      | Example Model |
| ----------------------- | ------------------------------------------------ | ------------- |
| Machine translation     | Converting text between languages                | Marian, T5    |
| Text summarization      | Creating concise summaries of longer texts       | BART, T5      |
| Data-to-text generation | Converting structured data into natural language | T5            |
| Grammar correction      | Fixing grammatical errors in text                | T5            |
| Question answering      | Generating answers based on context              | BART, T5      |

Representatives of this family of models include:

- [BART](https://huggingface.co/docs/transformers/model_doc/bart)
- [mBART](https://huggingface.co/docs/transformers/model_doc/mbart)
- [Marian](https://huggingface.co/docs/transformers/model_doc/marian)
- [T5](https://huggingface.co/docs/transformers/model_doc/t5)

## Choosing the right architecture

When working on a specific NLP task, how do you decide which architecture to use? Here's a quick guide:

| Task                                   | Suggested Architecture     | Examples      |
| -------------------------------------- | -------------------------- | ------------- |
| Text classification (sentiment, topic) | Encoder                    | BERT, RoBERTa |
| Text generation (creative writing)     | Decoder                    | GPT, LLaMA    |
| Translation                            | Encoder-Decoder            | T5, BART      |
| Summarization                          | Encoder-Decoder            | BART, T5      |
| Named entity recognition               | Encoder                    | BERT, RoBERTa |
| Question answering (extractive)        | Encoder                    | BERT, RoBERTa |
| Question answering (generative)        | Encoder-Decoder or Decoder | T5, GPT       |
| Conversational AI                      | Decoder                    | GPT, LLaMA    |

When in doubt about which model to use, consider:

1. What kind of understanding does your task need? (Bidirectional or unidirectional)
2. Are you generating new text or analyzing existing text?
3. Do you need to transform one sequence into another?

The answers to these questions will guide you toward the right architecture.

## Conclusion

In this section, we've explored the three main Transformer architectures and some specialized attention mechanisms. Understanding these architectural differences is crucial for selecting the right model for your specific NLP task.

As we move forward in the course, you'll get hands-on experience with these different architectures and learn how to fine-tune them for your specific needs. In the next section, we'll look at some of the limitations and biases present in these models that you should be aware of when deploying them.

<EditOnGithub source="https://github.com/huggingface/course/blob/main/chapters/en/chapter1/6.mdx" />

# Optional

- How 🤗 Transformers solve tasks: https://huggingface.co/learn/llm-course/chapter1/5
- Transformer Architectures: https://huggingface.co/learn/llm-course/chapter1/6
- Deep dive into Text Generation Inference with LLMs: https://huggingface.co/learn/llm-course/chapter1/8
- Bias and limitations: https://huggingface.co/learn/llm-course/chapter1/9

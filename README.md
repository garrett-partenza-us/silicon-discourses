# Silicon Discourses

> “It is not that we have a short space of time, but that we waste much of it.”
>
> -- <cite>Lucius Annaeus Seneca</cite>

Generating news articles, auto-completing your email greeting, unproductive chatbots -- these are the common use cases of large generative language models like GPT-3. What do they all have in common? They're all boring. Seriously, I would rather put a campfire out with my face than spend 2 months developing a chatbot that drives its users one step closer to digital retirement with each forward pass. 

So, what else can we make?

Well, a crude way of describing GPT-3's generative text capability is that somehow backpropagation over a swath of the internet's text has allowed its parameters to converge toward a general understanding of the framework by which humans communicate. While easy to monetize into small short-lived SaaS startups, I find this framework of communication banal, and the ability to generative more of it just the same.

One framework of communication I do find interesting, however, is that of the great Stoics. I've spent most mornings over the last three months reading the foundational writings; Discourses, Meditations, Letters from a Stoic, The Enchiridion, Hardship and Happiness. I find that their teachings on becoming comfortable with death, practicing temperance for leisure, and being unmoved by events out of our control, offer a practical dogma for obtaining tranquility. I also find that their call for toughness, both mental and physical, and the ability to endure hardship, fulfils a natural yearning inside me unnourished by modern pop culture. But there's one issue: there just isn’t enough. Most of the great Stoics either did not write themselves or had portions of writings lost in history. Plus, I assume we lost a few letters from Seneca when he was sentenced to death and all?  

So, rather than train GPT-3 on Wikipedia articles to make chatbots and unoriginal Saas startups that crowd your LinkedIn inbox, I'd rather train GPT-3 on Stoic philosophy to generate the lost books of Epictetus or gain a generalized understanding of Stoic philosophy as to generate new teachings. 

> “The happiness of your life depends upon the quality of your thoughts.”
>
> -- <cite>Marcus Aurelius</cite>

### The Trivial Solution

Ideally, we can just use the pretrained GPT-3 model to generate new Philosophy by priming it with Stoic prompts. After all, the best solution is always the [simplest](https://en.wikipedia.org/wiki/Occam%27s_razor). Unfortunately, doing so results in GPT-3 babbling on about Christian religion. Let's use Hugging Face to prompt GPT-3 with a fragment from Epictetus' Discourses, in hopes it will generate the remaining lost text.

Prompt: "Seek not that the things which happen should happen as you wish; but wish the things which happen to be as they are, and you will have a tranquil flow of life."

Output: "For so many are of one mind, and yet not all are of the same mind; there are, though in some there are some at liberty of mind, but very few that are able to discern what is really in their hearts. 1Jesis 2:28 ESV / 18 helpful votes Helpful Not Helpful "The living things which are in heaven, shall not prosper after their own offering, because heaven will not eat those things which are in heaven that are not in the earth. 2 Peter 4:11 ESV / 17 helpful votes Helpful Not Helpful "And to that which is above all things, you shall judge not; but only you shall be called in the name of the Father" (John 1:12). Romans 6:15 ESV / 17 helpful votes Helpful Not"

I’m not sure about you, but I don’t feel more enlightened after reading this.

### Starting from Scratch

I won’t go into extreme detail regarding my methodology, but I will give a general overview and also link some resources which I found helpful throughout development. 

Since using the pretrained model doesn’t work, I had to start from scratch. First, I rewrote the GPT-3 architecture model class in PyTorch (12 layers, 12 heads, 768 model dimension and 3072 hidden dimensions). This resulted in about 125 million parameters between the trainable embedding layer, generator class, and decoder architecture. I then downloaded books 1 through 4 of Epictetus' Discourses from the internet (as books 5 through 8 were lost in time, maybe we can generate some remanence from their fragments) and saved it as a text file. After preprocessing the text file, I trained a byte-pair encoding tokenizer with a vocabulary size of 5000 tokens using Google's SentencePiece Library. With the tokenizer, model class, and dataset, all that was left was to train. Using my graduate school's high-performance computing center, I wrote a training loop and trained the model on a NVIDIA P100 GPU for ~46 hours. 

### Results

As with GPT-3, we'll generate 100 tokens from the same prompt. 

Prompt: "Seek not that the things which happen should happen as you wish; but wish the things which happen to be as they are, and you will have a tranquil flow of life."

Output: "Formed when you do not even among them. And so? Formed by all things and when some and when you also into them as it is whether you who is a slave and so long and when it is not because also of the gods that it? It is not how it who are all that it? It pleases god. And who does not that it is it do not possible? It seemed that which do not takes itself. And they? Why do not"
 
While GPT-3's output was certainly better at handling long range dependencies and formulating semantically correct sentences, this output reads more like philosophy by asking questions and using verbiage similar to Epictetus. For a dataset that is only 581 kilobytes compared to GPT's 45 terabytes, a single GPU as compared to ten thousand GPUs, and zero dollars as compared to 5 million dollars, I’d say my model put up a fair fight.

For anyone interested, [here](https://raw.githubusercontent.com/garrett-partenza-us/silicon_discourses/main/silicon_discourses.txt) is a link to 10,000 tokens generated from the same prompt. I like to call it Silicon Discourses.

### Useful Resources
1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
2. [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
3. [GPT-3 Paper](https://www.google.com/search?q=GPT3+paper+archiv&oq=GPT3+paper+archiv&aqs=chrome..69i57j0i512l2j0i10i22i30j0i22i30.2412j0j4&sourceid=chrome&ie=UTF-8)

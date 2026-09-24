from pathlib import Path
import re

import pymupdf

fitz = pymupdf

base = Path(__file__).resolve().parent / "data"
base.mkdir(exist_ok=True)


def write_text_doc(filename: str, content: str):
    path = base / filename
    path.write_text(content, encoding="utf-8")


def write_pdf_doc(filename: str, text: str, pages: int = 2):
    path = base / filename
    doc = fitz.open()
    for page_index in range(pages):
        page = doc.new_page()
        y = 72
        lines = text.splitlines()
        for line in lines:
            page.insert_text((72, y), line, fontsize=10.5, fontname="helv")
            y += 15
            if y > 760:
                break
        if page_index < pages - 1:
            page = doc.new_page()
    doc.save(path)
    doc.close()


def build_text(prefix: str, body: str) -> str:
    return f"{prefix}\n\n" + body

texts = {
    "machine_learning.txt": build_text(
        "Machine learning is a subfield of artificial intelligence that focuses on building systems that can learn patterns from data and improve their performance without being explicitly programmed for every decision.",
        """Machine learning is a subfield of artificial intelligence that focuses on building systems that can learn patterns from data and improve their performance without being explicitly programmed for every decision. Instead of writing thousands of rules by hand, machine learning models identify structure in historical datasets and use that structure to make predictions, classify information, or discover patterns in new data. This approach has become central to modern software systems, powering recommendation engines, fraud detection, medical diagnostics, autonomous vehicles, and language applications.

At a high level, machine learning involves a learning loop. First, a dataset is collected and prepared. This may involve cleaning missing values, converting formats, normalizing numbers, and labeling examples for supervised learning tasks. Next, a model is chosen. Models can be simple, such as logistic regression or decision trees, or complex, such as random forests, support vector machines, gradient boosting machines, and neural networks. Then the model is trained on part of the data while holding aside a validation or test set to measure generalization. Training means adjusting model parameters so that the loss function decreases and the model becomes more accurate on the training examples.

One of the most important ideas in machine learning is the difference between fitting and generalization. A model that memorizes every training example may perform extremely well on the training data but poorly on new data. This problem is called overfitting. Generalization is the ability of a model to perform well on unseen examples, which is why machine learning practitioners carefully split data, tune hyperparameters, and use regularization techniques. Cross-validation helps estimate how a model will behave when exposed to real-world data, and feature engineering or representation learning can improve the quality of the learned signal.

Machine learning is usually divided into several categories. Supervised learning uses labeled examples, such as classifying emails as spam or predicting house prices based on features like square footage and neighborhood. Unsupervised learning works with unlabeled data and tries to discover structure, such as clustering customers into segments or identifying anomalies in manufacturing data. Semi-supervised learning combines small amounts of labeled data with large amounts of unlabeled data, which is common when labeling is expensive. Reinforcement learning trains agents through trial and error by rewarding desirable behavior, and it is widely used in robotics, game AI, and resource optimization.

Another important concept is feature engineering, which is the process of selecting or transforming raw data into representations that are easier for a model to learn from. Traditional machine learning pipelines often required domain expertise to create useful features. For example, in a credit-risk model, a developer may create features such as debt-to-income ratio, account age, and number of missed payments. In modern deep learning systems, representation learning can automatically extract meaningful features from raw inputs like images or text. This has reduced the need for hand-crafted features in many domains.

Evaluation is a critical part of any machine learning project. A model may appear effective on paper but fail in the real world if evaluation is done incorrectly. Data leakage is a common concern, where information from the test set influences the training process and leads to inflated performance estimates. Metrics matter as well. For classification tasks, common metrics include accuracy, precision, recall, F1-score, and ROC-AUC. For regression, metrics like mean squared error and mean absolute error are common. In production, businesses also track operational metrics such as latency, fairness, maintenance cost, and business impact.

Machine learning has also become deeply integrated with software engineering practices. Teams often build pipelines that automate data collection, preprocessing, training, evaluation, and deployment. MLOps brings together machine learning engineering with DevOps practices to make model delivery safer and more repeatable. This includes experiment tracking, versioning datasets and models, automated testing, and monitoring models after deployment. When deployed models drift away from the expected data distribution, they may produce poor predictions unless retraining or recalibration is introduced.

A strong understanding of machine learning helps professionals across many domains, including healthcare, finance, agriculture, manufacturing, and cybersecurity. For example, in healthcare, machine learning can help detect diseases from scans, stratify risk, and optimize hospital operations. In finance, it can detect suspicious transactions and predict future customer behavior. In manufacturing, machine learning models can predict equipment failures before they happen. These applications show that machine learning is not just a research topic; it is a practical engineering discipline with broad societal impact.

The field continues to evolve quickly. New research explores self-supervised learning, graph neural networks, trustworthy AI, and low-resource learning. These innovations widen the range of applications and improve the ability of models to learn from limited or noisy data. As the field grows, important questions remain around fairness, transparency, data ownership, privacy, and accountability. These issues are especially important when machine learning systems affect hiring, lending, health, education, and public policy. A responsible machine learning practitioner must think about the social consequences of their work, not only the model accuracy.

In summary, machine learning is the discipline of teaching computers to learn patterns from data. It combines statistics, optimization, software engineering, and domain knowledge. By understanding the core principles of training, evaluation, and generalization, decision-makers can use machine learning to solve meaningful problems and create systems that improve over time. Its true power lies in converting data into useful predictions and decisions that support humans in complex, real-world situations.""",
    ),
    "deep_learning.txt": build_text(
        "Deep learning is a branch of machine learning that uses artificial neural networks with many layers to model complicated patterns in data.",
        """Deep learning is a branch of machine learning that uses artificial neural networks with many layers to model complicated patterns in data. While traditional machine learning often relied on handcrafted features, deep learning systems learn useful representations automatically from raw inputs such as images, audio, and text. This ability to learn hierarchical features has made deep learning extremely powerful across a wide range of applications, including computer vision, speech recognition, robotics, healthcare, and natural language processing.

The basic unit of a deep learning system is the neuron or node, which receives inputs, applies a weighted transformation, and passes the result through a nonlinear activation function. In a neural network, neurons are organized into layers. The input layer receives the raw data, hidden layers transform the data through many mathematical operations, and the output layer produces the final prediction or classification. The depth of the network is what makes it deep: a network with multiple hidden layers can represent highly complex functions. This expressive power is one reason deep learning can outperform simpler models on challenging tasks.

A crucial concept in deep learning is backpropagation. During training, a network produces an output, compares it to the expected target using a loss function, and then computes gradients to determine how each weight should change. These gradients are propagated backward through the network so that the parameters can be updated using an optimization algorithm such as stochastic gradient descent or Adam. This process is repeated over many iterations using batches of training examples. Over time, the model learns patterns that reduce prediction error and improve generalization.

Deep learning architectures come in several forms depending on the type of data and the task. Convolutional neural networks (CNNs) are specialized for grid-like data such as images and are widely used in object detection, medical image analysis, and visual recognition. Recurrent neural networks (RNNs) and transformer-based models are designed for sequential data such as text, speech, and time series. Transformers have become especially influential in natural language processing because they can model long-range dependencies and process sequences in parallel, which makes training much more efficient than earlier recurrent approaches.

The field of computer vision has benefited enormously from deep learning. Models can classify images, detect objects, segment scenes, and recognize faces with impressive accuracy. In autonomous vehicles, deep learning helps interpret road scenes and detect obstacles. In healthcare, deep learning systems can analyze radiology scans to highlight suspicious regions or assist with disease diagnosis. In agriculture, vision models can monitor crop health and detect pests. These examples show that deep learning is not only a theoretical concept but also a practical technology that powers highly useful systems.

In the domain of audio and speech, deep learning models can transcribe speech, synthesize voices, and recognize emotions from tone. Speech recognition systems now convert spoken language into text with high accuracy, enabling virtual assistants and transcription tools. Generative models can also create realistic speech, making voice interfaces more natural and useful. This has created new applications in education, accessibility, customer service, and media production.

Deep learning has also transformed natural language processing. Transformer architectures such as BERT, GPT, and large language models are built on neural attention mechanisms that allow the model to focus on specific parts of an input sequence while generating context-aware interpretations. These models can perform tasks like summarization, translation, question answering, sentiment analysis, and coding assistance. The success of large language models has influenced many industries by automating tasks that previously required humans to read or draft text at scale.

Training deep learning models is computationally expensive. Large networks require significant memory and processing power, often using graphics processing units (GPUs) or specialized accelerators. The training pipeline may include data augmentation, regularization, and optimization strategies like weight decay, dropout, and learning rate scheduling. Even with these tools, models can still require careful tuning and long training runs. This has led to the rise of distributed training, cloud computing, and frameworks such as TensorFlow, PyTorch, and JAX.

Despite these successes, deep learning has limitations. Models can be opaque, making it difficult to explain why a particular prediction was made. They can also be brittle when applied to data that differs from the training distribution. Bias in the data can be learned and amplified by the model, leading to unfair decisions in sensitive contexts. Because deep learning models are data hungry, organizations must invest in quality data collection, labeling, and governance. Research into explainability, robustness, and fairness is therefore essential to building trustworthy AI systems.

The impact of deep learning extends beyond software. It influences scientific discovery, creative work, and public services. In biology, deep learning helps model protein structures and detect patterns in genetic data. In climate science, it can help analyze satellite imagery and improve forecasting. In education, adaptive learning systems can personalize lesson recommendations. As deep learning becomes more accessible through pretrained models and cloud APIs, more teams can build prototypes and deploy systems faster than before.

A good understanding of deep learning requires both conceptual and practical knowledge. It is not enough to know formulas or architecture names; one must also understand data quality, training dynamics, performance evaluation, and deployment concerns. Students and engineers who learn these principles can build models that are useful, efficient, and responsible. Modern AI increasingly depends on deep learning, yet the most successful systems still require careful design, testing, and ongoing refinement.

In summary, deep learning represents a powerful class of machine learning methods centered on multi-layer neural networks. By learning hierarchical features from raw data, these models can solve complex tasks across vision, language, speech, science, and industry. As research continues, deep learning will likely remain a core technology shaping the next generation of intelligent systems.""",
    ),
    "natural_language_processing.txt": build_text(
        "Natural language processing, or NLP, is a field of artificial intelligence focused on enabling computers to understand, interpret, and generate human language.",
        """Natural language processing, or NLP, is a field of artificial intelligence focused on enabling computers to understand, interpret, and generate human language. This includes language comprehension, language generation, translation, summarization, sentiment analysis, and dialogue systems. NLP has become one of the most visible areas of AI because it powers applications such as smart assistants, customer support chatbots, search engines, grammar checkers, and translation services. These systems are designed to work with the ambiguity, nuance, and context that naturally appear in human communication.

At its core, NLP is concerned with converting language into representations that computers can process. Early systems used rule-based approaches and hand-crafted grammars, which worked in narrow domains but struggled with variation and unpredictability in language. More modern NLP relies on statistical methods and neural networks to learn from large corpora of text. Models can estimate the probability of a word sequence, infer meaning from surrounding context, and produce coherent text in response to a prompt. This shift from symbolic methods to data-driven learning has greatly expanded the practical success of NLP.

One important concept in NLP is tokenization, which is the process of splitting text into smaller units such as words, subwords, or characters. Tokenization matters because it defines the basic vocabulary used by a model. Many modern systems rely on byte-pair encoding or WordPiece tokenization to handle large vocabularies efficiently and represent rare words more effectively. After tokenization, models often map tokens to embeddings, which are dense vectors that capture semantic relationships. A word like "bank" may have different meanings depending on the sentence, and contextual embeddings help resolve this ambiguity.

Context is critical in language understanding. A sentence may contain multiple possible interpretations depending on surrounding words, history, and tone. This is why transformer-based architectures became so important in NLP. The transformer model introduced attention, which allows a model to weigh the influence of different words when processing a sequence. Unlike older recurrent models, transformers process sequences in parallel and can capture long-range dependencies more effectively. This improvement enabled major breakthroughs in machine translation, text summarization, question answering, and sentence classification.

NLP tasks can be grouped into several broad categories. Text classification involves assigning labels to documents, such as spam detection or sentiment classification. Named entity recognition identifies people, locations, organizations, and other entities in text. Machine translation transforms one language into another while preserving meaning. Summarization compresses long documents into shorter summaries. Question answering involves retrieving or generating answers from text, while dialogue systems aim to hold meaningful conversations with users. These tasks are often evaluated using metrics such as accuracy, F1-score, BLEU, ROUGE, and human judgment.

Language models are a particularly influential part of modern NLP. A language model learns to predict the next token in a sequence, which allows it to generate fluent text. Large language models are trained on massive datasets and can perform many tasks with few examples or instructions. This capability is central to chatbots, writing assistants, coding tools, and retrieval systems. However, even powerful language models can produce incorrect or nonsensical outputs, so it is important to evaluate them carefully and use them with grounding or retrieval when factual accuracy matters.

NLP also supports information retrieval and search. Search systems use document indexing, ranking, and relevance scoring to retrieve the most useful results for a query. In modern systems, retrieval can be combined with generation to build RAG pipelines, where the model uses relevant passages from a knowledge base before answering. This approach reduces hallucinations and allows the system to answer questions using up-to-date or domain-specific information. Related tasks include semantic search, document clustering, and conversational search.

A key challenge in NLP is handling ambiguity, idioms, sarcasm, and domain-specific vocabulary. Human language is rich, noisy, and highly context-dependent. Systems must deal with spelling errors, code-switching, dialect differences, and incomplete sentences. In professional settings, domain-adapted models are often necessary because general-purpose language models may not understand specialized terminology in areas like law, medicine, or engineering. Fine-tuning and retrieval augmentation are common strategies for improving performance in such environments.

Data quality and linguistic diversity are essential for building strong NLP systems. Training corpora must be large, representative, and fairly balanced to avoid embedding harmful biases or underrepresenting certain communities. Ethical concerns in NLP include privacy, fairness, and misuse. For example, a system that generates harmful or deceptive text could be used for misinformation or abusive automation. Responsible NLP development requires transparent evaluation, safeguards, and human oversight, especially when models interact with users directly.

NLP has broad real-world impact. It powers translation systems that connect people across languages, assistive technologies for accessibility, clinical documentation tools, educational tutoring systems, and public-sector chatbots. In business, NLP helps analyze customer feedback, summarize support tickets, and automate repetitive writing tasks. In science, it can help analyze research papers, extract facts, and discover patterns across large literature corpora.

The future of NLP is likely to involve multimodal systems that combine text with images, audio, and structured data. These systems may reason across multiple modalities and interact with the world more naturally. Yet even as models become more capable, the fundamentals remain important: understanding language representations, context, evaluation, and human-centered design. NLP is not simply about making machines talk; it is about building systems that help people communicate, understand information, and make decisions with greater ease and precision.""",
    ),
    "generative_ai.txt": build_text(
        "Generative AI refers to machine learning systems that can create new content rather than only classify or predict existing examples.",
        """Generative AI refers to machine learning systems that can create new content rather than only classify or predict existing examples. This includes text generation, image generation, music composition, code synthesis, and even synthetic data creation. Generative models are built on statistical patterns learned from large datasets and can produce outputs that resemble the training distribution while still being novel. In recent years, generative AI has become one of the most discussed and commercially important areas of artificial intelligence, especially because it has become accessible to non-experts through user-friendly interfaces and APIs.

A fundamental idea behind generative AI is that a model learns the structure of data by observing many examples. For instance, a text model might learn the patterns of grammar, vocabulary, and discourse by reading billions of words. Once trained, it can continue a sentence, answer a question, or write a paragraph in a style that is consistent with the data it has seen. Similarly, image models learn visual patterns from labeled and unlabeled examples and can create new scenes, illustrations, or stylized artwork. These models do not store exact copies of the training examples; instead, they learn useful statistical relationships that enable generation across a broad space of possibilities.

There are several common families of generative models. Autoencoders learn compressed representations of data and can reconstruct or generate variations of input examples. Generative adversarial networks, or GANs, train two networks against each other: one generates content, and the other tries to distinguish real examples from generated ones. Variational autoencoders model latent spaces that allow smooth variation between examples. More recently, diffusion models have become highly successful in image generation because they learn to reverse a noisy process and iteratively reconstruct realistic visual content. For text, transformer-based autoregressive models such as GPT-like systems are widely used because they generate output one token at a time based on contextual input.

Generative AI has huge practical value. In software development, code generation tools can suggest functions, complete boilerplate, and explain errors. In marketing, content generation systems can draft blog posts, ad copy, and social media messages. In design, generative tools support rapid ideation by creating mood boards, layouts, and visual concepts. In education, AI systems can generate explanations, quizzes, and practice problems. In customer support, chatbots can draft responses, summarize conversations, and help agents answer questions faster. These benefits can increase productivity and make complex tasks more accessible to people with limited technical backgrounds.

However, generative AI also comes with limitations and risks. One major issue is hallucination: the model may produce information that sounds confident but is inaccurate or unsupported by facts. This is especially problematic in high-stakes domains such as healthcare, law, finance, or scientific research. Another issue is bias. If the training data contains stereotypes or underrepresentation, a generative model may reproduce or amplify those patterns. Privacy is also a concern when models are trained on personal data or when users input sensitive information into third-party tools. Safety concerns include misinformation, impersonation, spam, and the creation of deepfakes or manipulative content.

To mitigate these risks, many organizations use grounding techniques. Retrieval-augmented generation, or RAG, is one popular method in which a model is given relevant retrieved passages from a trusted knowledge base before answering a question. This improves factual grounding and allows the model to answer using sources rather than memory alone. Human review, clear usage policies, sandboxing, and evaluation frameworks are also important when deploying generative systems in production. Responsible AI teams often combine model evaluation with governance and compliance procedures to reduce risk.

Generative AI is also influencing the labor market and business strategies. Some tasks can be automated, while others require human oversight and domain expertise. For example, a support agent may use generative AI to draft an initial response, but a human should verify factual correctness and tone before sending it. In design workflows, a model may propose dozens of concepts quickly, but an illustrator or product designer still provides judgment on aesthetics and brand alignment. This shift does not eliminate the need for skilled workers; instead, it changes how work is performed and creates new opportunities for collaboration between humans and AI.

There are also important questions about intellectual property and copyright. If a model is trained on public web content or copyrighted material, there may be legal and ethical concerns about how the model learns and reproduces patterns from that data. Some organizations are exploring data licensing, consent-based training, and opt-out mechanisms. The legal landscape is still evolving, and governments, companies, and researchers are trying to balance innovation with rights protection. These policy discussions will shape how generative AI is developed and deployed in the future.

The performance of generative AI depends strongly on the quality of the model, the data, and the system design. Scaling up parameters and training data often improves capability, but it also increases cost, energy use, and deployment complexity. Researchers are actively investigating efficient architectures, smaller models, and multimodal systems that combine text, vision, and audio. This trend is especially important for organizations that need to deploy AI responsibly on limited hardware or within regulated environments.

Generative AI has moved from research laboratories into everyday products. It is used in writing assistants, product recommendation tools, question-answering systems, digital art platforms, and coding copilots. The accessibility of these tools is one reason the field has seen remarkable public attention. At the same time, the conversation around generative AI must remain grounded in evidence, accountability, and practical evaluation. The technology may be powerful, but it is only as trustworthy as the data, design choices, and safeguards that surround it.

In summary, generative AI is a set of methods for creating new data and content from learned patterns. It spans text, images, audio, and code, and it has already changed how people work, create, and search for information. As the field matures, the challenge is not just making models more capable, but making them accurate, safe, and useful in the real world. This balance between innovation and responsibility will define the next phase of generative AI.""",
    ),
    "artificial_intelligence.pdf": ""  # placeholder, replaced below
}

# Write TXT files
for filename, text in texts.items():
    if filename.endswith(".txt"):
        write_text_doc(filename, text)

# Create a real PDF with readable text using paragraph blocks
pdf_text = """Artificial intelligence is the discipline of creating systems that can perform tasks normally associated with human intelligence. These tasks include reasoning, perception, learning, language understanding, planning, and problem solving. The history of AI is often traced to early work in logic, symbolic reasoning, and mathematics, but the field gained major momentum in the twentieth century as computers became more powerful and data became more available. The goal of artificial intelligence is not simply to mimic human behavior, but to design systems that can solve problems in robust and scalable ways, often by learning from large amounts of information.

Modern AI includes several major paradigms. Rule-based systems encode expert knowledge as explicit logic for decision making, while statistical learning methods derive patterns from data. Symbolic AI focuses on reasoning with symbols, knowledge graphs, and logic rules. Connectionist approaches, especially neural networks, emphasize learning by adjusting many parameters across layers of computation. These different approaches illustrate that AI is not a single technique but a broad research area containing many methods that are useful for different problems.

One of the central ideas in AI is automation of decision making. In healthcare, AI systems can support diagnosis by analyzing medical images or patient records. In finance, they can detect fraud patterns, forecast demand, or optimize investment strategies. In logistics, they can predict route efficiency and warehouse demand. In education, adaptive systems can personalize learning content to different student needs. These applications show that AI is useful when it improves access to information, supports decisions, and reduces repetitive human effort.

Machine learning is a major subfield of AI that allows systems to improve with experience. Instead of explicitly programming all rules, developers provide data and define a learning objective. The system then searches for patterns in the data and uses those patterns to make predictions. This process requires a combination of data quality, model design, evaluation, and iteration. A strong AI system is not just a model; it is also a deployment pipeline that manages training, testing, drift detection, and monitoring.

Language is a particularly important domain in AI. Natural language processing systems try to understand text, speech, and meaning in context. Such systems can summarize long documents, translate between languages, extract information, and answer questions. AI-powered chat systems are becoming common because they can engage with users in natural, conversational ways. Yet these systems still require careful design to keep outputs reliable, transparent, and aligned with human values.

A major challenge in AI is balancing capability with reliability. Highly capable systems can produce useful results but also make incorrect or misleading statements. This problem is often called hallucination when a model presents unsupported information confidently. In fields like medicine, law, and scientific research, even a small error can have serious consequences. This is why AI practitioners emphasize evaluation, benchmarking, human oversight, and robust engineering practices.

Ethics and governance are now essential parts of AI development. Bias in training data can lead to biased predictions and unfair treatment. Data privacy is also a key concern when models use personal or sensitive information. Transparency matters because users and stakeholders deserve to know when AI influences important decisions. Accountability is especially important in high-risk settings, such as hiring, lending, policing, or government programs. Responsible AI requires not only technical competence but also social awareness and legal understanding.

The future of AI will likely involve more multimodal systems, better reasoning capabilities, and stronger human-AI collaboration. AI can support people by handling routine or repetitive tasks, surfacing insights from massive datasets, and helping experts make better decisions. It cannot replace human judgment entirely, especially in contexts that require empathy, ethics, and context-sensitive reasoning. The best AI systems therefore enhance human ability rather than replace it.

Artificial intelligence is both a technical science and a societal endeavor. It changes how organizations operate, how people interact with services, and how research is conducted across nearly every domain. As the field matures, it is crucial to develop systems that are accurate, explainable, fair, and beneficial to society. Those principles will guide the next generation of AI systems and determine whether they become trusted tools or risky technology.

This document explores the foundations of artificial intelligence, its practical applications, and the need for responsible development. The field connects mathematics, computer science, cognitive science, and engineering. It has created tools for automation, prediction, and discovery that were once considered impossible. Because AI systems can learn from data, they are increasingly useful in domains where uncertainty, complexity, and scale make traditional programming methods less effective.

In summary, artificial intelligence is a broad discipline with both technical depth and human impact. It brings together ideas from logic, statistics, optimization, and neuroscience. A thoughtful understanding of AI helps people make better decisions about how to design, deploy, and govern these systems. For students and professionals, AI is not only a technical topic but a strategic technology that will shape business, science, and society in the years ahead."""

pdf_path = base / "artificial_intelligence.pdf"
doc = fitz.open()
paragraphs = [p.strip() for p in pdf_text.split("\n\n") if p.strip()]
for paragraph in paragraphs:
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 540, 760), paragraph, fontsize=10.5, fontname="helv", align=3)

doc.save(pdf_path)
doc.close()

for filename in sorted(base.iterdir()):
    if filename.suffix.lower() == ".txt":
        wc = len(re.findall(r"\b\w+\b", filename.read_text(encoding="utf-8")))
        print(f"{filename.name}: {wc} words")

pdf_doc = fitz.open(pdf_path)
pdf_words = len(re.findall(r"\b\w+\b", " ".join(page.get_text() for page in pdf_doc)))
pdf_doc.close()
print(f"{pdf_path.name}: {pdf_words} words")

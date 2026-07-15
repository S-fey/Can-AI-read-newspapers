# Can AI Read Newspapers? 📰

A comprehensive research project investigating whether different AI models can accurately comprehend and extract information from newspaper articles.

## 🎯 Project Overview

This research project evaluates the capability of modern AI language models to read, understand, and answer questions about newspaper content. The evaluation methodology uses a rigorous "Gold Standard" approach where AI-generated answers are compared against verified human answers to quantify accuracy rates across different models.

The goal is to understand:
- **Model Performance**: How different AI models perform on newspaper comprehension tasks
- **Consistency**: Whether certain models are more reliable than others
- **Real-World Applicability**: Whether AI can be trusted to extract accurate information from news sources
- **Limitations**: What types of newspaper content or question types challenge AI models

## 📊 Methodology

### Gold Standard Evaluation Framework

This project employs a Gold Standard evaluation methodology:

1. **Source Material**: Real newspaper articles are selected as evaluation materials
2. **Question Generation**: Comprehension questions are created based on article content
3. **Human Baseline**: Expert human annotators provide correct answers, establishing the "Gold Standard"
4. **AI Evaluation**: Each AI model answers the same questions
5. **Comparison**: AI answers are compared against the human Gold Standard to determine accuracy

This approach ensures:
- **Objectivity**: Answers can be objectively verified against a baseline
- **Comparability**: Different models can be fairly compared using identical criteria
- **Reproducibility**: Results can be verified and validated by others
- **Quantifiability**: Accuracy metrics can be precisely calculated

## 🧠 AI Models Evaluated

This project evaluates performance across multiple AI models, including but not limited to:
- **GPT Series** (OpenAI models)
- **Claude Series** (Anthropic models)
- **Gemini/PaLM** (Google models)
- **LLaMA** (Meta models)
- **Specialized Models** (domain-specific or fine-tuned variants)

Each model is tested under identical conditions to ensure fair comparison.

## 📈 Evaluation Metrics

The project measures:

- **Exact Match Accuracy**: Percentage of answers that exactly match the Gold Standard
- **Semantic Similarity**: How closely AI answers align with the correct answer in meaning
- **Precision & Recall**: How well models capture required information
- **Error Analysis**: Types and patterns of mistakes made by each model
- **Confidence Scores**: Whether models express appropriate confidence in their answers

## 🏗️ Project Structure

```
Can-AI-read-newspapers/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   ├── newspaper_articles/            # Raw newspaper articles
│   ├── questions/                     # Evaluation questions
│   ├── gold_standard_answers/         # Human-verified correct answers
│   └── model_predictions/             # AI model responses
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Initial data analysis
│   ├── 02_model_evaluation.ipynb      # Evaluation pipeline
│   └── 03_results_analysis.ipynb      # Statistical analysis & visualization
├── src/
│   ├── data_loader.py                 # Load and preprocess data
│   ├── model_interface.py             # Unified API for querying models
│   ├── evaluator.py                   # Comparison and scoring logic
│   ├── metrics.py                     # Evaluation metrics
│   └── visualization.py               # Generate charts and reports
├── results/
│   ├── performance_metrics.json       # Aggregated results
│   ├── detailed_analysis.csv          # Per-example analysis
│   └── figures/                       # Visualizations and graphs
└── tests/
    ├── test_evaluator.py              # Unit tests
    └── test_metrics.py                # Metrics validation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for AI models you wish to evaluate (OpenAI, Anthropic, Google, etc.)
- 2GB+ of disk space for data and results

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/S-fey/Can-AI-read-newspapers.git
   cd Can-AI-read-newspapers
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

### Quick Start

1. **Load and explore the data**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Run model evaluation**:
   ```bash
   python -m src.evaluate --models gpt-4 claude-3 gemini-pro --batch-size 10
   ```

3. **Analyze results**:
   ```bash
   jupyter notebook notebooks/03_results_analysis.ipynb
   ```

## 🔍 Usage Examples

### Basic Model Evaluation

```python
from src.model_interface import ModelInterface
from src.evaluator import Evaluator

# Initialize models
gpt4 = ModelInterface('gpt-4')
claude = ModelInterface('claude-3-opus')

# Load evaluation data
questions, gold_answers = load_evaluation_data('data/')

# Get model responses
gpt4_responses = [gpt4.answer(q) for q in questions]
claude_responses = [claude.answer(q) for q in questions]

# Evaluate accuracy
evaluator = Evaluator(gold_answers)
gpt4_accuracy = evaluator.calculate_accuracy(gpt4_responses)
claude_accuracy = evaluator.calculate_accuracy(claude_responses)

print(f"GPT-4 Accuracy: {gpt4_accuracy:.2%}")
print(f"Claude Accuracy: {claude_accuracy:.2%}")
```

### Detailed Analysis

```python
from src.evaluator import Evaluator

evaluator = Evaluator(gold_answers)

# Get detailed error analysis
results = evaluator.evaluate_detailed(model_responses)

# Identify problematic categories
problematic_questions = results[results['is_correct'] == False]
print(problematic_questions.groupby('question_type').size())
```

## 📋 Key Features

✅ **Automated Evaluation Pipeline**: Streamlined process for testing multiple models  
✅ **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and more  
✅ **Error Analysis**: Detailed breakdown of failure modes  
✅ **Visualization Tools**: Generate publication-ready charts and graphs  
✅ **Reproducibility**: Fixed random seeds and documented procedures  
✅ **Scalability**: Process hundreds of articles and questions efficiently  
✅ **Model Agnostic**: Easily add new models to the evaluation framework  

## 📊 Results Summary

Key findings from the evaluation:

- **Model Performance Range**: Accuracy varies significantly across models
- **Task Difficulty**: Certain question types (e.g., numerical facts, inferences) prove more challenging
- **Consistency**: Some models show more stable performance across different article types
- **Domain Effects**: Performance may vary based on newspaper category (politics, science, business, etc.)

*For detailed results, see the [results/](results/) directory and [detailed analysis](notebooks/03_results_analysis.ipynb)*

## 🛠️ Core Components

### Data Loader (`src/data_loader.py`)
Handles loading and preprocessing of:
- Newspaper articles (from various sources)
- Evaluation questions
- Gold Standard answers
- Metadata (publication date, category, etc.)

### Model Interface (`src/model_interface.py`)
Unified API for querying different AI models:
- Abstracts model-specific APIs into consistent interface
- Handles rate limiting and error recovery
- Manages API costs tracking
- Supports streaming and batch processing

### Evaluator (`src/evaluator.py`)
Core evaluation logic:
- Compares AI responses against Gold Standard
- Calculates multiple metrics
- Performs error analysis
- Generates detailed reports

### Metrics (`src/metrics.py`)
Implements evaluation metrics:
- Exact Match accuracy
- Semantic similarity (BLEU, ROUGE, BERTScore)
- Strict and lenient scoring modes
- Custom metric implementations

## 📈 Advanced Usage

### Custom Metrics

```python
from src.metrics import Metrics

metrics = Metrics()

# Add custom metric
metrics.register_metric('custom_similarity', custom_similarity_function)

# Use in evaluation
score = metrics.evaluate(predicted_answer, gold_answer, metric='custom_similarity')
```

### Batch Processing with Progress Tracking

```python
from src.model_interface import ModelInterface
from tqdm import tqdm

model = ModelInterface('gpt-4')

responses = []
for question in tqdm(questions, desc="Evaluating"):
    response = model.answer(question)
    responses.append(response)
```

## 🔬 Research Findings & Insights

### Notable Observations

1. **Context Dependency**: Models perform better when given full article context vs. extracted snippets
2. **Question Complexity**: Simple factual questions yield higher accuracy than complex inference questions
3. **Article Length**: Longer articles may present challenges due to token limits and attention mechanisms
4. **Temporal Information**: Models struggle with articles containing time-sensitive or date-specific information
5. **Numerical Accuracy**: Extracting precise numbers from text remains challenging for most models

### Limitations & Future Work

- **Scale**: Current evaluation covers X articles with Y questions
- **Language**: Limited to English-language newspapers
- **Freshness**: Models' knowledge cutoffs may affect performance on recent news
- **Bias**: Potential biases in model training data may affect comprehension of certain topics

### Future Directions

- [ ] Multi-lingual newspaper comprehension evaluation
- [ ] Fine-tuning models specifically for newspaper reading
- [ ] Evaluation on visual newspaper content (OCR integration)
- [ ] Real-time news comprehension testing
- [ ] Domain-specific fine-tuning analysis
- [ ] Comparison with specialized information extraction models

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with clear commit messages
4. **Add tests** for new functionality
5. **Submit a Pull Request** with a description of changes

### Contribution Ideas

- Adding new models to evaluate
- Improving evaluation metrics
- Expanding the dataset
- Optimizing performance
- Enhancing documentation
- Creating visualizations

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{can_ai_read_newspapers_2024,
  author = {S-fey},
  title = {Can AI Read Newspapers? A Comprehensive AI Model Evaluation},
  url = {https://github.com/S-fey/Can-AI-read-newspapers},
  year = {2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**S-fey** - [GitHub Profile](https://github.com/S-fey)

## 📞 Support & Questions

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/S-fey/Can-AI-read-newspapers/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/S-fey/Can-AI-read-newspapers/discussions)

## 🙏 Acknowledgments

- AI model providers (OpenAI, Anthropic, Google, Meta) for API access
- Newspaper datasets and sources
- Human annotators for creating the Gold Standard
- Contributors and researchers in the AI evaluation space

## 📚 References & Related Work

- Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions for Machine Reading Comprehension of Text"
- Wang et al. (2019). "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding"
- Hendrickson et al. (2023). "What You Can Cram Into a Single Vector: Probing Sentence Embeddings for Linguistic Properties"

---

**Last Updated**: 2024  
**Status**: Active Development


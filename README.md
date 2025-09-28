# TestSelectorForIntegration
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

An intelligent, AI-powered tool for reducing test suites by finding semantic redundancy in mock-based tests.

---

## The Problem: The Hidden Cost of Mock-Based Testing

In modern microservice architectures, integration testing is critical but also slow and expensive. To get fast feedback, developers write hundreds of unit tests that use **mocks** to simulate external dependencies.

However, this approach has two major problems:
1.  `Functional Redundancy`: Many tests, while different, end up verifying the exact same interaction or behavior, bloating the test suite without adding value.
2.  `Signal vs. Noise`: The key interaction (the mock) is often buried in a sea of setup and assertion code, making it difficult for automated tools to understand what is actually being tested.

This leads to slower CI/CD pipelines and a growing difficulty in maintaining the test suite.

## The Solution: A Test Selector

**TestSelectorForIntegration** addresses this problem by treating mock declarations not as mere code, but as a **pure semantic signal** that describes an integration point.

The framework analyzes a test class, isolates these "signals," transforms them into numerical vectors (embeddings) using state-of-the-art AI models, and groups (clusters) semantically similar tests. Finally, it selects a representative subset of tests that covers the maximum number of unique interactions.

This allows you to:
- `Drastically reduce the number of tests` to be run.
- `Maintain high coverage of critical integration points (CIP)`.
- `Enable faster, more targeted integration testing`, especially when used in combination with the [Ubiquo](https://github.com/yurimonti/UbiquoStubIntegration) framework.

## Key Features

- `AI-Powered Intelligent Selection`: Utilizes a variety of code language models (CodeBERT, StarCoder2, etc.) for deep semantic analysis.
- `Multi-Language Support`: Analyzes tests written in Python, Java, C#, Go, Ruby, PHP, and JavaScript.
- `Automated and Optimized Clustering`: Autonomously finds the best clustering strategy to group tests meaningfully.
- `Prioritized Selection Strategy`: Ensures that unique tests (outliers) and representatives from every functional group are always included.
- `Actionable Outputs`: Generates both a JSON list of tests and a ready-to-run test file with non-selected methods commented out.
- `Self-Evaluation`: Automatically calculates the CIP metric to provide immediate feedback on the selection's effectiveness.

## Installation

To get started, clone the repository and install the dependencies.

```bash
# Clone the repository
git clone https://github.com/TiburzioFoglia/TestSelectorForIntegration.git
```

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage
The tool is run from the command line. The basic command structure is:

```bash
python main.py [--code-analyzer MODEL_NAME] [--full-tests] FILE_NAME [NUM_TESTS]
```
#### Arguments:
- `FILE_NAME (required)`: The path to the test class file to be analyzed.
- `NUM_TESTS (optional)`: The number of tests you wish to select. If not specified, the tool will select a default number.

#### Options:
- `--code-analyzer MODEL_NAME`: Specifies which AI model to use for generating embeddings.
Possible values: codeBert, graphCodeBert, codeT5, polyCoder, sentenceTransformer, unixCoder, starCoder2.
Default: codeBert.
- `--full-tests`: If present, the tool will analyze the entire body of the test methods. **Default (recommended)**: Analyzes only the lines of code containing mocks.

#### Examples:

**Example 1: Basic Analysis** 

Analyze `tests/my_test_class.py` and select 10 tests using the default model (codeBert) and the only mocks strategy.
```bash
python main.py tests/my_test_class.py 10
```

**Example 2: Using a Different Model**

Analyze the same file, but use the StarCoder2 model for higher accuracy.
```bash
python main.py --code-analyzer starCoder2 tests/my_test_class.py 10
```

**Example 3: Experimental Analysis (Full Tests)**

Run the analysis using the full body of the tests (for comparative purposes).
```bash
python main.py --full-tests tests/my_test_class.py 10
```
#### Outputs:

After execution, the tool will create:
- `extracted_methods.json`: An intermediate file with the extracted methods and code snippets.
- `selected_methods.json`: The final list of the names of the selected methods.
- `commented_FILE_NAME`: A copy of the input file with non-selected tests commented out, ready to be executed.

The final CIP score will also be printed to the console.

## Integration with Ubiquo

This project is designed to act as the intelligent selection "brain" for the **[Ubiquo](https://github.com/yurimonti/UbiquoStubIntegration)** testing framework.

The combined workflow is:
1.  **Selection (This Project):** `TestSelectorForIntegration` analyses `my_test_class.py` and generates the commented version of the file.
2.  **Transformation (Ubiquo):** `Ubiquo` receives the file and executes it in its "Integration Mode," transforming mock calls into real network calls.

This synergy enables targeted and efficient integration testing, making the automatic detection of "mock drift" practical even in fast CI/CD pipelines.

## Supported Technologies

#### Programming Languages
- `Python`
- `Java`
- `C#`
- `Go`
- `Ruby`
- `PHP`
- `JavaScript` 

#### AI Models
- `codeBert`
- `graphCodeBert`
- `codeT5`
- `polyCoder`
- `sentenceTransformer`
- `unixCoder`
- `starCoder2`

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.




















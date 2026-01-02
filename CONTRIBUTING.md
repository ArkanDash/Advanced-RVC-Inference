# Contributing to Advanced-RVC-Inference

Thank you for your interest in contributing to Advanced-RVC-Inference! We're excited to have you here and can't wait to see what you'll bring to our community. This guide will help you get started with contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Ways to Contribute](#ways-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

This project adheres to our [Terms of Use](README.md#terms-of-use). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or higher**
- **Git** for version control
- **CUDA-compatible GPU** (optional, for GPU-accelerated development)
- **FFmpeg** for audio processing

### Fork and Clone the Repository

1. Fork the repository on GitHub by clicking the "Fork" button
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/ArkanDash/Advanced-RVC-Inference.git
```

## Development Setup

### Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

### Install Development Dependencies

```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from source
pip install -e .
```

### Verify Installation

```bash
# Run a quick check to ensure everything is installed correctly
python -c "from advanced_rvc_inference import RVCInference; print('Installation successful!')"
```

## Ways to Contribute

### 1. Reporting Issues

Help us improve the project by reporting bugs and suggesting features:

- Search existing issues before creating a new one
- Use clear, descriptive titles
- Provide detailed information:
  - Steps to reproduce the bug
  - Expected vs. actual behavior
  - Error messages and logs
  - Environment details (OS, Python version, GPU if applicable)
- Include screenshots or code snippets when relevant

### 2. Suggesting Enhancements

We welcome ideas for new features! When suggesting an enhancement:

- Explain why the feature would be useful
- Describe potential use cases
- Provide implementation suggestions if possible
- Mention any alternatives you've considered

### 3. Writing Code

Areas where contributions are particularly welcome:

- **Core Inference Engine**: Optimizations, new features, bug fixes
- **Pitch Extraction Methods**: Adding or improving f0 estimators
- **Web UI**: Gradio interface improvements, new tabs, better UX
- **Documentation**: Tutorials, API docs, code comments
- **Testing**: Unit tests, integration tests, benchmark scripts

### 4. Improving Documentation

Help make the project more accessible:

- Fix typos and grammatical errors
- Improve code comments and docstrings
- Add examples and tutorials
- Translate documentation

## Coding Standards

### Python Style Guide

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some adaptations:

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Use descriptive variable and function names
- Write docstrings for all public functions and classes

### Code Organization

```
advanced_rvc_inference/
├── core/           # Core processing modules
├── tabs/           # UI tab components
├── library/        # ML libraries and utilities
│   ├── uvr5_lib/   # UVR separation library
│   └── speaker_diarization/  # Speaker diarization modules
├── infer/          # Inference engines
├── rvc/            # RVC-specific modules
│   └── realtime/   # Real-time processing
├── uvr/            # UVR separation modules
├── utils/          # Utility functions and variables
├── assets/         # Resource files
├── configs/        # Configuration files
├── gui.py          # Main Gradio interface
├── api.py          # Python API
├── cli.py          # Command-line interface
└── variables.py    # Global configuration
```

### Import Order

Organize imports in the following order:

1. Standard library imports
2. Third-party imports
3. Local application imports

### Type Hints

Use type hints for function signatures when possible:

```python
from typing import Optional, List

def process_audio(
    input_path: str,
    output_path: Optional[str] = None,
    pitch_change: int = 0
) -> str:
    """Process audio with voice conversion."""
    # Implementation
    return output_path
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_package.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=advanced_rvc_inference
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_feature_description.py`
- Follow the pattern: `test_<function_name>_<scenario>`
- Include docstrings for test functions
- Mock external dependencies when appropriate

### Test Coverage Areas

Priority areas for test coverage:

- Core inference functions
- Audio processing pipelines
- Configuration handling
- CLI commands
- API endpoints

## Submitting Changes

### Creating a Pull Request

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following the coding standards

4. **Commit your changes** with a clear message

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

### Pull Request Guidelines

- Keep PRs focused and small (one feature or fix per PR)
- Write a clear title and description
- Link related issues
- Include screenshots for UI changes
- Ensure all tests pass
- Request review from maintainers
- Address review feedback promptly

### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] I have tested the changes locally
- [ ] I have added/updated tests
- [ ] Tests pass successfully

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed self-review
- [ ] I have commented complex code
- [ ] My changes generate no new warnings
- [ ] I have updated documentation if needed
```

## Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

### Commit Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring (no behavior change) |
| `perf` | Performance improvements |
| `test` | Adding or modifying tests |
| `chore` | Maintenance tasks |

### Examples

```
feat(core): add support for new pitch extraction method

feat(gui): add batch processing tab

fix(api): resolve memory leak in inference pipeline

docs(readme): add Colab badge to quick start section

refactor(inference): improve audio loading performance

test(core): add unit tests for f0 extraction module
```

### Best Practices

- Use imperative mood ("add feature" not "added feature")
- Keep the subject line under 72 characters
- Capitalize the first letter of the description
- Use the body to explain "what" and "why", not "how"

## Community

Join our community for discussions and support:

- **Discord**: [https://discord.gg/hvmsukmBHE](https://discord.gg/hvmsukmBHE)
- **GitHub Issues**: [https://github.com/ArkanDash/Advanced-RVC-Inference/issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)
- **GitHub Discussions**: [https://github.com/ArkanDash/Advanced-RVC-Inference/discussions](https://github.com/ArkanDash/Advanced-RVC-Inference/discussions)

## Recognition

Contributors will be recognized in:

- The [README.md](README.md#credits) credits section
- The project's release notes
- Our community channels

---

Thank you for contributing to Advanced-RVC-Inference! Your efforts help make this project better for everyone.

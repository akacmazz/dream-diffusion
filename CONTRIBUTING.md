# Contributing to DREAM Diffusion

Thank you for your interest in contributing to DREAM Diffusion! This document provides guidelines for contributing to the project.

## 🤝 Ways to Contribute

- **Bug Reports**: Report issues or bugs you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes or new features
- **Documentation**: Improve documentation and examples
- **Testing**: Help test the code on different platforms

## 🚀 Getting Started

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/dream-diffusion.git
   cd dream-diffusion
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks** (optional but recommended)
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, readable code
   - Add comments for complex logic
   - Follow the existing code style

3. **Test your changes**
   ```bash
   pytest tests/
   python -m flake8 src/
   python -m black src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style

### Python Style Guide
- Follow [PEP 8](https://pep8.org/) guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where appropriate

### Commit Message Format
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples:**
```bash
feat(training): add DREAM loss implementation
fix(data): resolve CelebA dataset loading issue
docs(readme): update installation instructions
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when necessary

### Test Structure
```python
def test_feature_name():
    """Test description."""
    # Arrange
    setup_test_data()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected_value
```

## 📚 Documentation

### Docstring Format
Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description of the function.
    
    Longer description if needed.
    
    Args:
        param1: Description of parameter 1.
        param2: Description of parameter 2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: Description of when this exception is raised.
    """
    pass
```

### Documentation Updates
- Update README.md for significant changes
- Add docstrings to new functions and classes
- Update type hints
- Include examples for new features

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**:
   - Python version
   - PyTorch version
   - GPU type and memory
   - Operating system

2. **Steps to Reproduce**:
   - Minimal code example
   - Input data description
   - Expected vs actual behavior

3. **Error Messages**:
   - Full error traceback
   - Log files if available

### Bug Report Template
```markdown
**Environment:**
- Python: 3.8.10
- PyTorch: 2.0.1
- GPU: RTX 3070 (8GB)
- OS: Ubuntu 20.04

**Description:**
Brief description of the bug.

**Steps to Reproduce:**
1. Run this code...
2. With this data...
3. Observe the error...

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens.

**Error Message:**
```
Full error traceback here
```
```

## ✨ Feature Requests

When suggesting features:

1. **Use Case**: Describe why this feature would be useful
2. **Implementation**: Suggest how it might be implemented
3. **Examples**: Provide usage examples
4. **Alternatives**: Consider alternative solutions

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts

### PR Description Template
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Existing tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## 🔍 Code Review Process

1. **Automated Checks**: CI/CD runs tests and linters
2. **Maintainer Review**: Project maintainers review code
3. **Community Feedback**: Other contributors may provide input
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to main branch

## 🎯 Development Priorities

Current focus areas:
1. **Training Stability**: Improving crash protection and recovery
2. **Memory Optimization**: Reducing GPU memory usage
3. **Evaluation Metrics**: Adding more comprehensive evaluation
4. **Documentation**: Improving examples and tutorials
5. **Platform Support**: Better Colab/Kaggle integration

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and community discussions
- **Discord/Slack**: [Link to community chat] (if available)

## 🏆 Recognition

Contributors will be:
- Listed in the README.md
- Mentioned in release notes
- Added to the CONTRIBUTORS.md file

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DREAM Diffusion! 🚀
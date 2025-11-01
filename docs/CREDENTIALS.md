# Credential Management Guide

## Overview

This project uses environment-based configuration files to manage credentials securely.

## Security Rules

**NEVER commit actual `.env` files to Git!**

- ✅ Commit: `.env.example`, `.env.dev.example`, etc.
- ❌ Never commit: `.env`, `.env.dev`, `.env.prod`, etc.

## Setup Instructions

### 1. For Development

Copy the example file and add your credentials:

```bash
cp .env.dev.example .env
```

Then edit `.env` and add your actual API key:

```bash
OPENAI_API_KEY=sk-your-actual-key-here
ENVIRONMENT=dev
DEBUG=True
LOG_LEVEL=DEBUG
```

### 2. For Staging

```bash
cp .env.staging.example .env.staging
# Edit .env.staging with staging credentials
```

### 3. For Production

```bash
cp .env.prod.example .env.prod
# Edit .env.prod with production credentials
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | `sk-proj-...` |
| `ENVIRONMENT` | Current environment | `dev`, `staging`, `prod` |
| `DEBUG` | Enable debug mode | `True`, `False` |
| `LOG_LEVEL` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Getting Your OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (you won't be able to see it again!)
5. Paste it into your `.env` file

## Loading Environment Variables in Code

### Option 1: python-dotenv (Recommended)

Install:
```bash
pip install python-dotenv
```

Usage in Python:
```python
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
api_key = os.getenv("OPENAI_API_KEY")
```

### Option 2: Export in Terminal (Temporary)

```bash
export OPENAI_API_KEY='your-key-here'
python app.py
```

## Best Practices

1. **Use Different Keys for Different Environments**
   - Dev: Free tier or test API key
   - Prod: Paid tier with rate limits

2. **Rotate Keys Regularly**
   - Change API keys every 90 days
   - Immediately rotate if compromised

3. **Never Hardcode Credentials**
   ```python
   # ❌ BAD
   api_key = "sk-1234567890"
   
   # ✅ GOOD
   api_key = os.getenv("OPENAI_API_KEY")
   ```

4. **Use Secrets Management in Production**
   - AWS Secrets Manager
   - Azure Key Vault
   - HashiCorp Vault
   - GitHub Secrets (for CI/CD)

5. **Check Before Committing**
   ```bash
   # See what you're about to commit
   git status
   git diff
   
   # Make sure .env is not listed!
   ```

## Troubleshooting

### "OPENAI_API_KEY not found"

1. Check `.env` file exists in project root
2. Verify the key is correctly formatted (no quotes needed in .env)
3. Make sure `load_dotenv()` is called before accessing the variable
4. Try printing it: `print(os.getenv("OPENAI_API_KEY"))`

### "Command not found: python"

Use the full path to your virtual environment Python:
```bash
.venv/bin/python app.py
```

### Key Compromised?

1. Immediately delete the key from OpenAI dashboard
2. Create a new key
3. Update all `.env` files
4. Check if key was committed to Git (if yes, rotate immediately!)

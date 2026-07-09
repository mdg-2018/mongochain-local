"""Helper script for setting up mongochain in Google Colab.

Run this in your first Colab cell:

!pip install pymongo openai anthropic google-generativeai
!pip install git+https://github.com/robin-mongodb/mongochain.git

Embeddings are handled by MongoDB Atlas Automated Embedding (`autoEmbed`);
no Voyage AI client-side install or API key is required.
"""


def setup_colab():
    """Install required dependencies for Google Colab."""
    import subprocess
    import sys

    packages = [
        "pymongo",
        "openai",
        "anthropic",
        "google-generativeai",
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("\n✅ All dependencies installed!")
    print("You can now import mongochain:")
    print("  from mongochain import MongoAgent")


def verify_setup():
    """Verify that all dependencies are properly installed."""
    issues = []

    try:
        import pymongo
        print(f"✅ pymongo {pymongo.__version__}")
    except ImportError:
        issues.append("pymongo")

    try:
        import openai
        print(f"✅ openai {openai.__version__}")
    except ImportError:
        issues.append("openai")

    try:
        import anthropic
        print(f"✅ anthropic {anthropic.__version__}")
    except ImportError:
        issues.append("anthropic")

    try:
        import google.generativeai
        print("✅ google-generativeai installed")
    except ImportError:
        issues.append("google-generativeai")

    if issues:
        print(f"\n❌ Missing packages: {', '.join(issues)}")
        print("Run: pip install " + " ".join(issues))
        return False

    print("\n✅ All dependencies verified!")
    return True


if __name__ == "__main__":
    setup_colab()
    verify_setup()

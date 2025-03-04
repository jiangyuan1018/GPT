# generate_text.py
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from gpt.generate import main

if __name__ == "__main__":
    main()
